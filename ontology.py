from owlready2 import *
import nlp


def populate_ontology(w2v_model, input_onto, output_onto):
    """
    Функция загружает онтологию в память, извлекает из нее индивидов
    классов и на их основе ищет потенциальных индивидов.
    """
    world = World()
    onto = world.get_ontology(input_onto).load()

    classes = onto.classes()

    for c in classes:
        instances = [nlp.get_name_from_IRI(inst)
                     for inst in onto.get_instances_of(c)]

        # Удаляем индивиды, если их нет в модели.
        for inst in instances:
            if inst not in w2v_model.wv.vocab.keys():
                instances.remove(inst)

        similar = []
        if(instances != []):
            similar = w2v_model.wv.most_similar(positive=instances,
                                                topn=10)

        print(str(c).split(".")[1])
        for sim in similar:
            print("{}: {:.4f}".format(sim[0], sim[1]))
        print()

        for s in similar:
            # Сохранение найденных экземпляров класса.
            eval("onto." + str(c).split(".")[1] + "(s[0])")

    onto.save(file=output_onto)
