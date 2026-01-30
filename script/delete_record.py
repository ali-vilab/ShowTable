import os
import json


def delete_success_record(success_record_file, key, value):
    with open(success_record_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    new_data = []
    for d in data:
        if d['stage'] == 'success' and d[key] == value:
            continue
        new_data.append(d)

    os.rename(success_record_file, success_record_file + '.bak')

    with open(success_record_file, 'w') as f:
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def delete_record(record_file, step_key, step_values):
    with open(record_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    new_data = []
    for d in data:
        steps = d['steps']
        invalid_flag = False
        for step in steps:
            # if step['role'].startswith(step_key) and step['content'] in step_values:
            # if step['role'] == 'image_gen' and not os.path.exists(os.path.join(gt_root, step['content'])):
            if step['role'].startswith('edit_') and not os.path.exists(os.path.join(gt_root, step['content'])):
                invalid_flag = True
                break
        if invalid_flag:
            continue
        new_data.append(d)
    print(len(data) - len(new_data))
    os.rename(record_file, record_file + '.bak')

    with open(record_file, 'w') as f:
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
