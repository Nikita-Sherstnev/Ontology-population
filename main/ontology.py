from owlready2 import *
import nlp


def find_candidate_instances(w2v_vectors, tagged_words, input_onto, topn):
    candidate_instances = defaultdict(list)

    world = World()
    onto = world.get_ontology(input_onto).load()

    onto_classes = onto.classes()

    for onto_class in onto_classes:
        instances = [nlp.get_name_from_IRI(inst)
                     for inst in onto.get_instances_of(onto_class)]

        for inst in instances:
            if inst not in w2v_vectors.vocab.keys():
                instances.remove(inst)

        similar = find_by_cos_similarity(
            w2v_vectors, instances, onto_class, topn)

        similar = filter_by_pos(similar, instances, tagged_words)

        for s in similar[:]:
            if s[1] <= 0.4:
                similar.remove(s)

        candidate_instances[onto_class] = similar

    return candidate_instances


def find_by_cos_similarity(w2v_vectors, instances, onto_class, topn):
    similar = list()
    if(instances != []):
        similar = w2v_vectors.most_similar(positive=instances,
                                           topn=topn)
    return similar


def filter_by_pos(similar, instances, tagged_words):
    tags = []
    for inst in instances:
        tags.append(tagged_words[inst])

    for s in similar[:]:
        if s[0] in tagged_words:
            if tagged_words[s[0]] not in tags:
                similar.remove(s)

    return similar


def populate_ontology(candidate_instances, input_onto, output_onto):
    world = World()

    onto = world.get_ontology(input_onto).load()

    for onto_class, instances in candidate_instances.items():

        print_class_and_similar(onto_class, instances)

        for inst in instances:
            _save_instance(onto, onto_class, inst)

    onto.save(file=output_onto)


def _save_instance(onto, onto_class, inst):
    eval("onto." + str(onto_class).split(".")[1] + "(inst[0])")


def calculate_metrics(candidate_instances, correct_instances):
    matches_counter = 0
    new_instances_counter = 0
    correct_instances_amount = len(correct_instances)
    correct_instances = correct_instances.copy()

    for _, instances in candidate_instances.items():
        for inst in instances:
            new_instances_counter += 1

            if inst[0] in correct_instances[:]:
                matches_counter += 1
                correct_instances.remove(inst[0])

    print('Экземпляров в тексте: ',  correct_instances_amount)
    print('Всего новых экземляров: ', str(new_instances_counter))
    print('Правильно извлеченные экземпляры: ', matches_counter)
    print('Полнота: ', matches_counter/correct_instances_amount)
    print('Точность: ', matches_counter/new_instances_counter)


def print_class_and_similar(onto_class, similar):
    print(str(onto_class).split(".")[1])
    for sim in similar:
        print("{}: {:.4f}".format(sim[0], sim[1]))

    print()
