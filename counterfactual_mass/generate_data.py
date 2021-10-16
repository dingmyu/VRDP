import json
import os

root_dir = '../data/'

result_dict = {}
for process_index in range(100):
    gt = json.load(open(f'{root_dir}causal_mass/sim_{process_index:05d}/annotations/annotation.json'))
    objects = os.listdir(f'{root_dir}counterfactual_mass/sim_{process_index:05d}/')
    collisions = []
    for item in gt['collisions']:
        collision_frame = int(item['step'] / 7)
        if collision_frame < 125:
            collisions.append(item['object_idxs'])
    change_dict = {}
    for index in range(6):
        if f'mass_{index}_5' in objects:
            change_dict[index] = {
                'new': [],
                'old': [],
                'same': [],
            }
            counter_gt = json.load(open(f'{root_dir}counterfactual_mass/sim_{process_index:05d}/mass_{index}_5/annotations/annotation.json'))
            counter_collisions = []
            for item in counter_gt['collisions']:
                collision_frame = int(item['step'] / 7)
                if collision_frame < 125:
                    counter_collisions.append(item['object_idxs'])
            for co_item in counter_collisions:
                if co_item not in collisions and co_item not in change_dict[index]['new']:
                    change_dict[index]['new'].append(co_item)
            for co_item in collisions:
                if co_item not in counter_collisions:
                    if co_item not in change_dict[index]['old']:
                        change_dict[index]['old'].append(co_item)
                else:
                    if co_item not in change_dict[index]['same']:
                        change_dict[index]['same'].append(co_item)
    print(process_index, change_dict)
    result_dict[process_index] = {}
    for index, sub_dict in change_dict.items():
        if len(sub_dict['new']) or len(sub_dict['old']):
            result_dict[process_index][index] = {}
            if len(sub_dict['same']):
                result_dict[process_index][index]['exist'] = [sub_dict['same'][0]]
                result_dict[process_index][index]['exist'].extend(sub_dict['new'])
            else:
                result_dict[process_index][index]['exist'] = sub_dict['new']
            result_dict[process_index][index]['non_exist'] = sub_dict['old']
json.dump(result_dict, open('counterfactual_data_dict.json', 'w'))
