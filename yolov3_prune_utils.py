from collections.__init__ import namedtuple
from typing import Callable

from torch import nn as nn

from pruner.core.domain_registry import init_domain_core_pytorch
from pruner.domains.ioslicerbar.pytorch.propagator import simple_domain_propagate_o2io, simple_domain_propagate_o2i, \
    ConcatPropagateWrapper, simple_domain_propagate_o2o
from pruner.prunedirector.pytorch.dependency import PruneTracker, init_dep_node, add_dep_dsts, prune_module_domain, \
    regen_replace_modules_in_model
from pruner.util.pytorch.module_traverse import get_submodule, get_module_comp_rec


module_factory_fn, domain_factory_fn = init_domain_core_pytorch('ioslicerbar')

# utility for instrumentation
# out_module_names is a list of list, each inner most list is either one output from seq
# or all the outputs going into a shortcut

# end of instrumentation utility

in_out_record = namedtuple('in_out_record', ['in_module_name', 'out_module_names', 'op'])


def create_entries_for_conv_group(module, tracker:PruneTracker, tracker_str_stack):
    entry_module = None
    exit_module = None

    # check
    entry_module_entity = None
    exit_module_entity = None

    prev_module = None
    prev_module_entity = None
    for submodule_name, submodule_entity in get_submodule(module).items():
        cur_submodule_type  = type(submodule_entity)
        local_name = submodule_name
        global_name = tracker_str_stack + local_name
        if cur_submodule_type!= nn.Conv2d and cur_submodule_type!=nn.BatchNorm2d:
            assert cur_submodule_type == nn.LeakyReLU
            continue
        if entry_module is None:
            entry_module = global_name
            entry_module_entity = submodule_entity
        if cur_submodule_type == nn.Conv2d or cur_submodule_type == nn.BatchNorm2d:
            init_dep_node(tracker, global_name, submodule_entity, domain_factory_fn)
        if prev_module is not None:
            assert cur_submodule_type==nn.BatchNorm2d and type(prev_module_entity) == nn.Conv2d, 'unexpected relationship between conv layers'
            add_dep_dsts(tracker, prev_module,global_name,simple_domain_propagate_o2io)

        prev_module = global_name
        prev_module_entity = submodule_entity

        exit_module = global_name
        exit_module_entity = submodule_entity
    assert (type(exit_module_entity) == nn.BatchNorm2d and type(entry_module_entity) == nn.Conv2d) \
           or (entry_module==exit_module  and type(entry_module_entity) == nn.Conv2d), 'unexpected'
    return entry_module,exit_module


def link_prev_io_entry_to_current_module(tracker:PruneTracker,
                                         prev_out_entry:in_out_record,
                                         cur_in_entry,
                                         model):
    prev_out_groups = prev_out_entry.out_module_names
    if type(prev_out_entry.op) == nn.Sequential:
        assert(len(prev_out_groups)) ==1 and 1<=len(prev_out_groups[0])<=2, 'I thought we only had at most two groups'
        # there is one conv and one batchnorm, we just make us the dest of the conv
        for prev_out_group_member in prev_out_groups[0]:
            prev_out_group_member_module = get_module_comp_rec(model,prev_out_group_member)
            if type(prev_out_group_member_module) == nn.Conv2d:
                add_dep_dsts(tracker,prev_out_group_member,cur_in_entry,simple_domain_propagate_o2i)
    else:
        if prev_out_entry.op == 'shortcut':
            assert (len(prev_out_groups)) == 1, 'I thought output from shortcut never concat with others'
            # every body in the group has this as the dest
            for prev_out_group_member in prev_out_groups[0]:
                prev_out_group_member_module = get_module_comp_rec(model, prev_out_group_member)
                if type(prev_out_group_member_module) == nn.Conv2d:
                    add_dep_dsts(tracker, prev_out_group_member, cur_in_entry, simple_domain_propagate_o2i)

        elif prev_out_entry.op == 'route':
            assert ( 0 < len(prev_out_groups)) <= 2, 'I thought route always concat 1 or 2 '
            # we just need to have one faning out to me
            for prev_out_group_member in prev_out_groups[0]:
                prev_out_group_member_module = get_module_comp_rec(model, prev_out_group_member)
                if type(prev_out_group_member_module) == nn.Conv2d:
                    add_dep_dsts(tracker,prev_out_group_member,cur_in_entry,simple_domain_propagate_o2i)
            sibling_domain = domain_factory_fn(get_module_comp_rec(model,list(prev_out_groups[0])[0]))
            if len(prev_out_groups) == 2:
                for prev_out_group_member in prev_out_groups[1]:
                    prev_out_group_member_module = get_module_comp_rec(model, prev_out_group_member)
                    if type(prev_out_group_member_module) == nn.Conv2d:
                        add_dep_dsts(tracker, prev_out_group_member, cur_in_entry,
                                     ConcatPropagateWrapper(sibling_domain))
        else:
            assert False, 'unrecognize possibility'

def find_dst_with_prop(tracker:PruneTracker, src:str, prop:Callable, prop_cmp_neg=False):
    dsts=[]
    dst_fns = []

    for pair in zip(tracker.dependency[src].dsts, tracker.dependency[src].dst_fns):
        cur_dst, cur_dst_fn = pair[0],pair[1]
        if (prop_cmp_neg and cur_dst_fn != prop) or (not prop_cmp_neg and cur_dst_fn == prop):
            dsts.append(cur_dst)
            dst_fns.append(cur_dst_fn)
    return dsts,dst_fns


def link_siblings_children(tracker:PruneTracker):
    # for every src, if it is fanning out to some dst with o2o
    # it is fanning out to that person's dst with o2i
    for src in tracker.dependency.keys():
        siblings, _ = find_dst_with_prop(tracker,src,simple_domain_propagate_o2o)
        for sibling in siblings:
            sibling_children, children_props = find_dst_with_prop(tracker,sibling,simple_domain_propagate_o2o, prop_cmp_neg=True)
            #for sibling_child, sibling_child_prop in zip(sibling_children, children_props):
            tracker.dependency[src].dsts.extend(sibling_children)
            tracker.dependency[src].dst_fns.extend(children_props)

def link_shortcut_siblings(tracker:PruneTracker, sibling_set, model):
    for cur_member in sibling_set:
        # for every member, we add every other member
        # if member is batchnorm we dont add
        # if it is conv2d,  and sibling is conv2d, we add o2o, sibling is batchnorm, we add o2io
        cur_member_module = get_module_comp_rec(model, cur_member)
        if type(cur_member_module) == nn.BatchNorm2d:
            continue
        elif type(cur_member_module) == nn.Conv2d:
            current_dsts = tracker.get_dst_module_name(cur_member)
            for sibling in sibling_set:
                if sibling == cur_member or sibling in current_dsts:
                    continue
                # if the the sibling is batchnorm , we do o2io
                # if the sibling is conv, we do o2o
                sibling_module = get_module_comp_rec(model,sibling)
                if type(sibling_module) == nn.Conv2d:
                    add_dep_dsts(tracker, cur_member, sibling, simple_domain_propagate_o2o)
                elif type(sibling_module) == nn.BatchNorm2d:
                    add_dep_dsts(tracker, cur_member, sibling, simple_domain_propagate_o2io)
                else:
                    assert False, 'unexpected sibling'
        else:
            assert False, 'only conv2d and batchnorm2d is allowed'


def prune_segment_from_layer(tracker:PruneTracker, layer_name, segment, model):
    updated_module_name_set = prune_module_domain(tracker, layer_name, [segment])
    regen_replace_modules_in_model(tracker, updated_module_name_set, module_factory_fn, model)




