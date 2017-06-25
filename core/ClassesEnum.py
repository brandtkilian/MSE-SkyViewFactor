from enum import IntEnum


class Classes(IntEnum):
    """Enumerator to describes classes"""
    SKY = 0
    VEGETATION = 1
    BUILT = 2
    VOID = 3

    @staticmethod
    def nb_lbl(torify):
        return 3 if torify else 4
