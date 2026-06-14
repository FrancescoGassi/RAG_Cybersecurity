from __future__ import annotations


class BODMASFeatureNames:
    BYTE_HISTOGRAM_FEATURES = [
        f"byte_histogram_{i}" for i in range(256)
    ]
    BYTE_ENTROPY_FEATURES = [
        f"byte_entropy_{i}" for i in range(256)
    ]
    STRING_FEATURES = [
        f"string_{i}" for i in range(104)
    ]
    GENERAL_FILE_FEATURES = [
        f"general_file_{i}" for i in range(10)
    ]
    HEADER_FEATURES = [
        f"header_{i}" for i in range(62)
    ]
    SECTION_FEATURES = [
        f"section_{i}" for i in range(255)
    ]
    IMPORT_FEATURES = [
        f"import_{i}" for i in range(1280)
    ]
    EXPORT_FEATURES = [
        f"export_{i}" for i in range(128)
    ]
    DATADIR_FEATURES = [
        f"datadir_{i}" for i in range(30)
    ]

    @classmethod
    def get_all_feature_names(cls) -> list[str]:
        return (
            cls.BYTE_HISTOGRAM_FEATURES
            + cls.BYTE_ENTROPY_FEATURES
            + cls.STRING_FEATURES
            + cls.GENERAL_FILE_FEATURES
            + cls.HEADER_FEATURES
            + cls.SECTION_FEATURES
            + cls.IMPORT_FEATURES
            + cls.EXPORT_FEATURES
            + cls.DATADIR_FEATURES
        )

    @classmethod
    def get_feature_descriptions(cls) -> dict[str, str]:
        descriptions: dict[str, str] = {}

        for index, name in enumerate(cls.BYTE_HISTOGRAM_FEATURES):
            descriptions[name] = (
                f"relative frequency of byte 0x{index:02X}"
            )

        for index, name in enumerate(cls.BYTE_ENTROPY_FEATURES):
            descriptions[name] = (
                f"joint byte-entropy statistic for byte 0x{index:02X}"
            )

        for index, name in enumerate(cls.STRING_FEATURES):
            descriptions[name] = (
                f"printable-string statistic {index}"
            )

        for index, name in enumerate(cls.GENERAL_FILE_FEATURES):
            descriptions[name] = (
                f"general PE file statistic {index}"
            )

        for index, name in enumerate(cls.HEADER_FEATURES):
            descriptions[name] = (
                f"COFF or Optional Header field {index}"
            )

        for index, name in enumerate(cls.SECTION_FEATURES):
            descriptions[name] = (
                f"PE section statistic {index}"
            )

        for index, name in enumerate(cls.IMPORT_FEATURES):
            descriptions[name] = (
                f"import table feature {index}"
            )

        for index, name in enumerate(cls.EXPORT_FEATURES):
            descriptions[name] = (
                f"export table feature {index}"
            )

        for index, name in enumerate(cls.DATADIR_FEATURES):
            descriptions[name] = (
                f"PE data-directory feature {index}"
            )

        return descriptions