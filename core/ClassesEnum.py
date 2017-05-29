from enum import IntEnum


class Classes(IntEnum):
    SKY = 1
    VEGETATION = 2
    BUILT = 3
    VOID = 4

    @staticmethod
    def nb_lbl(torify):
        return 3 if torify else 4
