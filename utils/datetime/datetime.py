def yearMonth_to_index(year, month):
    """convert year, month to data index.

    Parameters
    ----------
    year : int
        year, from 1983.
    month : int
        month, from 1 to 12
    """
    gap = (year - 1983) * 12 + month - 1
    return gap

def index_to_yearMonth(index):
    """convert index to year, month

    Parameters
    ----------
    index : int
        data index.
    """
    year = index // 12 + 1983
    month = index % 12 + 1
    return (year, month)