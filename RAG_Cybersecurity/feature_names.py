from typing import List, Dict

class BODMASFeatureNames:
    """
    Nomi simbolici delle 2381 feature estratte dalla libreria LIEF
    per l'analisi statica di file Windows PE.
    I gruppi seguono l'ordine definito nel paper OLIVANDER.
    """

    BYTE_HISTOGRAM_FEATURES = [f"byte_histogram_{i}" for i in range(256)]
    BYTE_ENTROPY_FEATURES = [f"byte_entropy_{i}" for i in range(256)]
    STRING_FEATURES = [f"string_{i}" for i in range(104)]
    GENERAL_FILE_FEATURES = [f"general_file_{i}" for i in range(10)]
    HEADER_FEATURES = [f"header_{i}" for i in range(62)]
    SECTION_FEATURES = [f"section_{i}" for i in range(255)]
    IMPORT_FEATURES = [f"import_{i}" for i in range(1280)]
    EXPORT_FEATURES = [f"export_{i}" for i in range(128)]
    DATADIR_FEATURES = [f"datadir_{i}" for i in range(30)]

    @classmethod
    def get_all_feature_names(cls) -> List[str]:
        all_features = []
        all_features.extend(cls.BYTE_HISTOGRAM_FEATURES)
        all_features.extend(cls.BYTE_ENTROPY_FEATURES)
        all_features.extend(cls.STRING_FEATURES)
        all_features.extend(cls.GENERAL_FILE_FEATURES)
        all_features.extend(cls.HEADER_FEATURES)
        all_features.extend(cls.SECTION_FEATURES)
        all_features.extend(cls.IMPORT_FEATURES)
        all_features.extend(cls.EXPORT_FEATURES)
        all_features.extend(cls.DATADIR_FEATURES)
        return all_features

    @classmethod
    def get_feature_descriptions(cls) -> Dict[str, str]:
        desc = {}
        for i, name in enumerate(cls.BYTE_HISTOGRAM_FEATURES):
            desc[name] = f"Frequenza relativa del byte 0x{i:02X} nel file PE"
        for i, name in enumerate(cls.BYTE_ENTROPY_FEATURES):
            desc[name] = f"Distribuzione congiunta entropia-byte per 0x{i:02X}"
        for i, name in enumerate(cls.STRING_FEATURES):
            desc[name] = f"Statistica {i} sulle stringhe stampabili"
        for i, name in enumerate(cls.GENERAL_FILE_FEATURES):
            desc[name] = f"Informazione generale {i} del file PE (es. dimensione, flag header)"
        for i, name in enumerate(cls.HEADER_FEATURES):
            desc[name] = f"Campo {i} dell'header COFF/Optional Header"
        for i, name in enumerate(cls.SECTION_FEATURES):
            desc[name] = f"Attributo {i} dell'header di sezione"
        for i, name in enumerate(cls.IMPORT_FEATURES):
            desc[name] = f"Entry {i} della import address table"
        for i, name in enumerate(cls.EXPORT_FEATURES):
            desc[name] = f"Entry {i} della export table"
        for i, name in enumerate(cls.DATADIR_FEATURES):
            desc[name] = f"Entry {i} della data directory"
        return desc