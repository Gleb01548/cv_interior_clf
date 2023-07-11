import pandas as pd


def processing_df(input: str, output: str):
    """
    Метод меняет название столбцов, заполняет пропуски
    """
    df = pd.read_excel("объекты.xlsx", dtype="str")
    df.columns = [
        "image",
        "sofa",
        "wardrobe",
        "chair",
        "armchair",
        "table",
        "commode",
        "bed",
    ]
    df["image"] = df["image"].apply(
        lambda x: x + ".png" if not x.endswith(".png") else x
    )
    df.fillna("0", inplace=True)
    df.to_csv(output, index=False)
