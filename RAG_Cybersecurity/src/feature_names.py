from typing import List, Dict

class BODMASFeatureNames:
    """
    Nomi simbolici delle 2381 feature estratte dalla libreria LIEF
    per l'analisi statica di file Windows PE.
    I gruppi seguono l'ordine definito nel paper OLIVANDER.
    """

    # 1. Byte histogram (256 feature)
    # Misura la frequenza relativa di ciascun valore di byte nel file.
    BYTE_HISTOGRAM_FEATURES = [f"byte_histogram_{i}" for i in range(256)]

    # 2. Byte entropy histogram (256 feature)
    # Approssima la distribuzione congiunta p(H, X) di entropia H e valore di byte X.
    BYTE_ENTROPY_FEATURES = [f"byte_entropy_{i}" for i in range(256)]

    # 3. String information (104 feature)
    # Statistiche sulle stringhe stampabili presenti nel file (lunghezza, URL, registry key, ecc.).
    STRING_FEATURES = [f"string_{i}" for i in range(104)]

    # 4. General file (10 feature)
    # Dimensione del file e informazioni di base estratte dall'header PE.
    GENERAL_FILE_FEATURES = [f"general_file_{i}" for i in range(10)]

    # 5. Header information (62 feature)
    # Dati estratti dall'header COFF (timestamp, macchina target) e dall'Optional Header.
    HEADER_FEATURES = [f"header_{i}" for i in range(62)]

    # 6. Section information (255 feature)
    # Dati registrati nell'header di sezione (nome, dimensione, entropia, virtual size).
    SECTION_FEATURES = [f"section_{i}" for i in range(255)]

    # 7. Import information (1280 feature)
    # Informazioni sulle funzioni importate e le librerie associate (import address table).
    IMPORT_FEATURES = [f"import_{i}" for i in range(1280)]

    # 8. Export information (128 feature)
    # Lista delle funzioni esportate dal file PE.
    EXPORT_FEATURES = [f"export_{i}" for i in range(128)]

    # 9. Data directory information (30 feature)
    # Valori di size e virtual size di tutte le entry della data directory del file PE.
    DATADIR_FEATURES = [f"datadir_{i}" for i in range(30)]

    @classmethod
    def get_all_feature_names(cls) -> List[str]:
        """
        Restituisce la lista completa dei 2381 nomi simbolici delle feature,
        nell'ordine definito dalla libreria LIEF.
        """
        all_features = []
        all_features.extend(cls.BYTE_HISTOGRAM_FEATURES)   # 256
        all_features.extend(cls.BYTE_ENTROPY_FEATURES)      # 256
        all_features.extend(cls.STRING_FEATURES)             # 104
        all_features.extend(cls.GENERAL_FILE_FEATURES)       #  10
        all_features.extend(cls.HEADER_FEATURES)             #  62
        all_features.extend(cls.SECTION_FEATURES)            # 255
        all_features.extend(cls.IMPORT_FEATURES)             # 1280
        all_features.extend(cls.EXPORT_FEATURES)             # 128
        all_features.extend(cls.DATADIR_FEATURES)            #  30
        # Totale: 256+256+104+10+62+255+1280+128+30 = 2381
        return all_features

    @classmethod
    def get_feature_descriptions(cls) -> Dict[str, str]:
        """
        Restituisce un dizionario {nome_feature: descrizione} per le feature principali.
        """
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