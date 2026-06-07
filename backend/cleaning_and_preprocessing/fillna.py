from fastapi import HTTPException


class ConstantFiller:

    def fill(self, df, value, columns=None):
        if columns:

            for col in columns:

                if col not in df.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{col}' not found"
                    )

                df[col] = df[col].fillna(value)

        else:
            df = df.fillna(value)

        return df


def fillna_constant_data(df, value, columns=None):
    return ConstantFiller().fill(df, value, columns)