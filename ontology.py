from owlready2 import *
import nlp


def populate_ontology(w2v_model, tagged_words, input_onto, output_onto, topn):
    """
    Функция загружает онтологию в память, извлекает из нее индивидов
    классов и на их основе ищет потенциальных индивидов.
    """
    world = World()
    onto = world.get_ontology(input_onto).load()

    classes = onto.classes()

    for onto_class in classes:
        instances = [nlp.get_name_from_IRI(inst)
                     for inst in onto.get_instances_of(onto_class)]

        # Удаляем индивиды, если их нет в модели.
        for inst in instances:
            if inst not in w2v_model.wv.vocab.keys():
                instances.remove(inst)

        similar = []
        if(instances != []):
            similar = w2v_model.wv.most_similar(positive=instances,
                                                topn=topn)

        tags = []
        for inst in instances:
            tags.append(tagged_words[inst])

        for s in similar:
            if tagged_words[s[0]] not in tags:
                similar.remove(s)

        new_similar = []
        for s in similar:
            if s[1] >= 0.5:
                new_similar.append(s)

        print_class_and_similar(onto_class, new_similar, tagged_words)

        for s in new_similar:
            # Сохранение найденных экземпляров класса.
            eval("onto." + str(onto_class).split(".")[1] + "(s[0])")

    onto.save(file=output_onto)


def print_class_and_similar(onto_class, similar, tagged_words):
    print(str(onto_class).split(".")[1])
    for sim in similar:
        print("{}: {:.4f} {}".format(sim[0], sim[1], tagged_words[sim[0]]))
    print()
