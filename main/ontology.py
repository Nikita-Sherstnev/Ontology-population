from owlready2 import *
import nlp


def populate_ontology(w2v_vectors, tagged_words, correct_instances, input_onto, output_onto, topn):
    """
    Функция загружает онтологию в память, извлекает из нее индивидов
    классов и на их основе ищет потенциальных индивидов.
    """
    world = World()
    matches = 0
    correct_instances_amount = len(correct_instances)
    onto = world.get_ontology(input_onto).load()

    correct_instances = correct_instances.copy()

    onto_classes = onto.classes()

    new_instances_counter = 0
    for onto_class in onto_classes:
        instances = [nlp.get_name_from_IRI(inst)
                     for inst in onto.get_instances_of(onto_class)]

        # Удаляем индивиды, если их нет в модели.
        for inst in instances:
            if inst not in w2v_vectors.vocab.keys():
                instances.remove(inst)

        similar = []
        if(instances != []):
            similar = w2v_vectors.most_similar(positive=instances,
                                               topn=topn)

        tags = []
        for inst in instances:
            tags.append(tagged_words[inst])

        for s in similar[:]:
            if s[0] in tagged_words:
                if tagged_words[s[0]] not in tags:
                    similar.remove(s)

        for s in similar[:]:
            if s[1] <= 0.5:
                similar.remove(s)

        print_class_and_similar(onto_class, similar, tagged_words)

        for s in similar:
            # Если индивидуум класса найден правильно,
            # увеличиваем счетчик.
            if s[0] in correct_instances:
                correct_instances.remove(s[0])
                matches += 1

            new_instances_counter += 1
            # Сохранение найденных экземпляров класса.
            eval("onto." + str(onto_class).split(".")[1] + "(s[0])")

    print("Всего новых экземляров: ", str(new_instances_counter))
    onto.save(file=output_onto)
    print('Экземпляров в тексте: ', correct_instances_amount)
    print('Правильно извлеченные экземпляры: ', matches)
    print('Полнота: ', matches/correct_instances_amount)
    print('Точность: ', matches/new_instances_counter)


def print_class_and_similar(onto_class, similar, tagged_words):
    print(str(onto_class).split(".")[1])
    for sim in similar:
        print("{}: {:.4f}".format(sim[0], sim[1]), end=' ')
        if sim[0] in tagged_words:
            print(tagged_words[sim[0]])
        else:
            print()

    print()
