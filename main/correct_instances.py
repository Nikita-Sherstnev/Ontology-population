
def parse_instances_from_txt(correct_instances_file):
    with open(correct_instances_file, 'r', encoding='utf-8') as f:
        instances = f.readlines()

    instances = [x.lower().rstrip().replace(' ', '_') for x in instances]
    return instances
