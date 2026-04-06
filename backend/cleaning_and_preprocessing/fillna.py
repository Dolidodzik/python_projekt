def fillna_constant_data(df, value, columns=None):
    df = df.copy()

    if columns:
        if len(columns) == 0:
            return df

        for col in columns:
            df.loc[:, col] = df[col].fillna(value)
    else:
        df = df.fillna(value)

    return df.reset_index(drop=True)