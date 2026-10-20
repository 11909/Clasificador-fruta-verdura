from src.mapping import cargar_mapping, etiqueta_de, normalizar

MAPPING = cargar_mapping()


def test_normalizar_quita_numero_final():
    assert normalizar("Apple Granny Smith 2") == "apple granny smith"
    assert normalizar("Tomato_Cherry_Red") == "tomato cherry red"


def test_criterio_botanico():
    assert etiqueta_de("Tomato Cherry Red 1", MAPPING) == "fruta"
    assert etiqueta_de("Cucumber 3", MAPPING) == "fruta"
    assert etiqueta_de("Carrot 1", MAPPING) == "verdura"
    assert etiqueta_de("Potato Sweet 1", MAPPING) == "verdura"
    assert etiqueta_de("Mushroom champignon 1", MAPPING) == "excluir"


def test_coincidencia_por_palabra_completa():
    # "pepino" no debe confundirse con "pepper" ni "nut" con otras clases
    assert etiqueta_de("Pepino 1", MAPPING) == "fruta"
    assert etiqueta_de("Pepper Red 1", MAPPING) == "fruta"
    assert etiqueta_de("Nutmeg", MAPPING) is None


def test_etiquetas_validas():
    assert set(MAPPING.values()) <= {"fruta", "verdura", "excluir"}
