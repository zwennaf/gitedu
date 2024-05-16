# replace string values in columns with integer values
def PersonalStringtoInt(frame):
    # find columns with holiday data
    pers_cols = [col for col in frame.columns if "pers" in col]
    string_map = {
        "not once": 0,
        "one time": 1,
        "1 time": 1,
        "2 times": 2,
        "two times": 2,
        "3 times": 3,
        "three times": 3,
        "4 times": 4,
        "four times": 4,
        "5 times or more": 5,
        "five times or more": 5,
    }
    for key, value in string_map.items():
        for col in pers_cols:
            frame[col] = frame[col].replace(key, value)
    return frame


# extract only the columns with a given prefix from a dataframe
def FilterPrefix(frame, prefix):
    col1 = "nomem_encr"
    columns = [col for col in frame if col.startswith(prefix)]
    columns.insert(0, col1)
    return frame.filter(columns, axis=1)


# casts nomem_encr column to int
def IDtoInt(df):
    df["nomem_encr"] = df["nomem_encr"].astype(int)
    return df


def PrefixToNumeric(df):
    # df = df.fillna(method='bfill', axis=0).fillna(0)
    temp = df.drop(["nomem_encr"], axis=1)
    # temp = temp.dropna()
    for column in temp:
        if (
                column.startswith("life_sat")
                or column.startswith("open")
                or column.startswith("conscious")
                or column.startswith("extraV")
                or column.startswith("agree")
                or column.startswith("neuro")
                or column.startswith("pos")
                or column.startswith("neg")
                or column.startswith("optim")
                or column.startswith("pessim")
        ):
            df[column] = temp[column].str.extract("(\\d+)").astype(float, errors="ignore")
    return df


def extraToInt(df):
    extrav = [col for col in df.columns if "extraV" in col]
    # extrav = df[['extraV1', 'extraV2', 'extraV3', 'extraV4']]
    string_map = {
        "very inaccurate": 1,
        "moderately inaccurate": 2,
        "neither inaccurate nor accurate": 3,
        "moderately accurate": 4,
        "very accurate": 5,
    }
    for key, value in string_map.items():
        for col in extrav:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def agreetoInt(df):
    agree = [col for col in df.columns if "agree" in col]
    # agree = df[['agreeAb1', 'agreeAb2', 'agreeAb3', 'agreeAb4']]
    string_map = {
        "very inaccurate": 1,
        "moderately inaccurate": 2,
        "neither inaccurate nor accurate": 3,
        "moderately accurate": 4,
        "very accurate": 5,
    }
    for key, value in string_map.items():
        for col in agree:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def consToInt(df):
    conscious = [col for col in df.columns if "conscious" in col]
    # conscious = df[['conscious1', 'conscious2', 'conscious3', 'conscious4']]
    string_map = {
        "very inaccurate": 1,
        "moderately inaccurate": 2,
        "neither inaccurate nor accurate": 3,
        "moderately accurate": 4,
        "very accurate": 5,
    }
    for key, value in string_map.items():
        for col in conscious:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def neuroToInt(df):
    neurotic = [col for col in df.columns if "neurotic" in col]
    # neurotic = df[['neurotic1', 'neurotic2', 'neurotic3', 'neurotic4']]
    string_map = {
        "very inaccurate": 1,
        "moderately inaccurate": 2,
        "neither inaccurate nor accurate": 3,
        "moderately accurate": 4,
        "very accurate": 5,
    }
    for key, value in string_map.items():
        for col in neurotic:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def openToInt(df):
    open = [col for col in df.columns if "open" in col]
    # open = df[['open1', 'open2', 'open3', 'open4']]
    string_map = {
        "very inaccurate": 1,
        "moderately inaccurate": 2,
        "neither inaccurate nor accurate": 3,
        "moderately accurate": 4,
        "very accurate": 5,
    }
    for key, value in string_map.items():
        for col in open:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def PersonalityTraitScale(df):
    extraVertColumns = []
    for item in range(4):
        extraVertColumns.append("extraV" + str(item + 1))
    for ReversedColumns in ["extraV2", "extraV4"]:
        df[ReversedColumns] = (df[ReversedColumns] * -1) + 6
    df["ExtraVert"] = df.loc[:, extraVertColumns].sum(axis=1)
    # df["ExtraVert"] = (df["ExtraVert"] - df["ExtraVert"].min()) / (
    #         df["ExtraVert"].max() - df["ExtraVert"].min()
    # )

    agreeablecols = []
    for item in range(4):
        agreeablecols.append("agreeAb" + str(item + 1))
    for ReversedColumns in ["agreeAb1", "agreeAb3"]:
        df[ReversedColumns] = (df[ReversedColumns] * -1) + 6
    df["Agreeable"] = df.loc[:, agreeablecols].sum(axis=1)
    # df["Agreeable"] = (df["Agreeable"] - df["Agreeable"].min()) / (
    #         df["Agreeable"].max() - df["Agreeable"].min()
    # )

    consciouscolumns = []
    for item in range(4):
        consciouscolumns.append("conscious" + str(item + 1))
    for ReversedColumns in ["conscious2", "conscious4"]:
        df[ReversedColumns] = (df[ReversedColumns] * -1) + 6
    df["Conscious"] = df.loc[:, consciouscolumns].sum(axis=1)
    # df["Conscious"] = (df["Conscious"] - df["Conscious"].min()) / (
    #         df["Conscious"].max() - df["Conscious"].min()
    # )

    neuroticColumns = []
    for item in range(4):
        neuroticColumns.append("neurotic" + str(item + 1))
    for ReversedColumns in ["neurotic2", "neurotic4"]:
        df[ReversedColumns] = (df[ReversedColumns] * -1) + 6
    df["Neurotic"] = df.loc[:, neuroticColumns].sum(axis=1)
    # df["Neurotic"] = (df["Neurotic"] - df["Neurotic"].min()) / (
    #         df["Neurotic"].max() - df["Neurotic"].min()
    # )

    opennessColumns = []
    for item in range(4):
        opennessColumns.append("open" + str(item + 1))
    for ReversedColumns in ["open2", "open4"]:
        df[ReversedColumns] = (df[ReversedColumns] * -1) + 6
    df["Openness"] = df.loc[:, opennessColumns].sum(axis=1)
    # df["Openness"] = (df["Openness"] - df["Openness"].min()) / (
    #         df["Openness"].max() - df["Openness"].min()
    # )
    return df


def OptimToInt(df):
    optim = [col for col in df.columns if "optim" in col]
    string_map = {
        "strongly disagree": 1,
        "disagree": 2,
        "neutral": 3,
        "agree": 4,
        "strongly agree": 5,
    }
    for key, value in string_map.items():
        for col in optim:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def PessimToInt(df):
    pessim = [col for col in df.columns if "pessim" in col]
    string_map = {
        "strongly disagree": 1,
        "disagree": 2,
        "neutral": 3,
        "agree": 4,
        "strongly agree": 5,
    }
    for key, value in string_map.items():
        for col in pessim:
            df[col] = df[col].replace(key, value).astype(str)
    return df


def OptimPessim(df):
    optimCols = []
    for item in range(4):
        optimCols.append("optim_" + str(item + 1))
    df["Optimism"] = df.loc[:, optimCols].sum(axis=1)
    # df["Optimism"] = (df["Optimism"] - df["Optimism"].min()) / (
    #         df["Optimism"].max() - df["Optimism"].min()
    # )

    pessimCols = []
    for item in range(4):
        pessimCols.append("pessim_" + str(item + 1))
    df["Pessimism"] = df.loc[:, pessimCols].sum(axis=1)
    # df["Pessimism"] = (df["Pessimism"] - df["Pessimism"].min()) / (
    #         df["Pessimism"].max() - df["Pessimism"].min()
    # )
    return df


def NegaffectScale(df):
    neg_affect = []
    for item in range(10):
        neg_affect.append("neg_" + str(item + 1))
    df["NegAff"] = df.loc[:, neg_affect].sum(axis=1)
    # df["NegAff"] = (df["NegAff"] - df["NegAff"].min()) / (
    #         df["NegAff"].max() - df["NegAff"].min()
    # )
    return df


def PosAffectScale(df):
    neg_affect = []
    for item in range(10):
        neg_affect.append("pos_" + str(item + 1))
    df["PosAff"] = df.loc[:, neg_affect].sum(axis=1)
    # df["PosAff"] = (df["PosAff"] - df["PosAff"].min()) / (
    #         df["PosAff"].max() - df["PosAff"].min()
    # )
    return df


# ----------------------------------
# methods here

# extract a list of columns from df as a new df
def extractCols(df, cols, prefix=""):
    full_cols = [prefix + sub for sub in cols]
    full_cols.insert(0, 'nomem_encr')
    return df.filter(full_cols, axis=1)


def extractPersonalityCols(df, prefix):
    personality_cols = (
        "nomem_encr",  # ID
        "014", "015", "016", "017", "018",  # last SWLS 5-item life_satisfaction
        "020", "025", "030", "035",  # Extraversion
        "021", "026", "031", "036",  # Agreeableness
        "022", "027", "032", "037",  # Conscientiousness
        "023", "028", "068", "038",  # Neuroticism
        "024", "029", "034", "039",  # Openness
        "146",
        "147",
        "148",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",  # PANAS
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",  # PANAS
        "198",
        "201",
        "202",
        "207",
        # Optimism
        "200",
        "204",
        "205",
        "206",
    )  # Pessimism
    return extractCols(df, personality_cols, prefix=prefix)


def extractSocial(df, prefix):
    soc_int = ('277', '177')
    return extractCols(df, soc_int, prefix=prefix)


def extractBackgroundCols(df):
    bg_cols = ('geslacht', 'leeftijd')
    return extractCols(df, bg_cols)



# # life satisfaction score is calculated as the sum of its component variables
# # https://fetzer.org/sites/default/files/images/stories/pdf/selfmeasures/SATISFACTION-SatisfactionWithLife.pdf
# def CalculateLifeSatScore(df):
#     cols = []
#     for p in range(5):
#         cols.append("life_sat" + str(p + 1))
#     df['LS Score'] = df.loc[:, cols].sum(axis=1)
#     # scaling between 0 and 1
#     # OldMax = 35
#     # OldMin = 5
#     # OldRange = (OldMax - OldMin)
#     # df['LS Score'] = ((df['LS Score'] - OldMin) / OldRange)
#     return df


# removes the suffix at the end of each column name
# Note: meant for numeric suffixes that indicate year
def RemoveSuffix(frame):
    for col in frame.columns:
        if col != 'nomem_encr':
            c = col.split('_')
            c.pop()
            name = "_".join([item for item in c])
            frame = frame.rename(columns={col: name})
    return frame


# add the given suffix at the end of each column except nomem_encr
def AddSuffix(frame, suffix):
    for col in frame.columns:
        if col != 'nomem_encr':
            frame = frame.rename(columns={col: col + '_' + str(suffix)})
    return frame


def map_agree_categories(category):
    if 'strongly disagree' == category:
        return 1
    elif 'disagree' == category:
        return 2
    elif 'neutral' == category:
        return 3
    elif 'strongly agree' == category:
        return 5
    elif 'agree' == category:
        return 4
    else:
        return category


# %%
def map_education_categories(category):
    if category == 'wo (university)':
        return 6
    elif category == 'hbo (higher vocational education, US: college)':
        return 5
    elif category == 'mbo (intermediate vocational education, US: junior college)':
        return 4
    elif category == 'havo/vwo (higher secondary education/preparatory university education, US: senio':
        return 3
    elif category == 'vmbo (intermediate secondary education, US: junior high school)':
        return 2
    elif category == 'primary school':
        return 1
    else:
        return category


# %%
def map_urban_categories(category):
    if category == 'Extremely urban':
        return 5
    elif category == 'Very urban':
        return 4
    elif category == 'Moderately urban':
        return 3
    elif category == 'Slightly urban':
        return 2
    elif category == 'Not urban':
        return 1
    else:
        return category


# %%
def map_health_categories(category):
    if category == 'poor':
        return 1
    elif category == 'moderate':
        return 2
    elif category == 'good':
        return 3
    elif category == 'very good':
        return 4
    elif category == 'excellent':
        return 5
    else:
        return category


# %%
def map_depressed_categories(category):
    if category == 'never':
        return 1
    elif category == 'seldom':
        return 2
    elif category == 'sometimes':
        return 3
    elif category == 'often':
        return 4
    elif category == 'mostly':
        return 5
    elif category == 'continuously':
        return 6
    else:
        return category

# %%
def map_BIG_V_categories(category):  # Takes like 15 sec
    if category == 'very inaccurate':
        return 1
    elif category == 'moderately inaccurate':
        return 2
    elif category == 'neither inaccurate nor accurate':
        return 3
    elif category == 'moderately accurate':
        return 4
    elif category == 'very accurate':
        return 5
    else:
        return category


# %%
def map_self_esteem_categories(category):
    if '1 totally disagree' == category:
        return 1
    elif '7 totally agree' == category:
        return 7
    else:
        return category


# %%
def map_optimism_categories(category):
    if 'strongly disagree' == category:
        return 1
    elif 'disagree' == category:
        return 2
    elif 'neutral' == category:
        return 3
    elif 'strongly agree' == category:
        return 5
    elif 'agree' == category:
        return 4
    else:
        return category


# %%
def map_trust_others_categories(category):
    if category == 'I dont know':
        return None
    elif category == '0 You cant be too careful':
        return 0
    elif category == '10 Most people can be trusted':
        return 10
    else:
        return category


# %%
def map_open_mind_categories(category):
    if '1 extremely unimportant' == category:
        return 1
    elif '7 extremely important' == category:
        return 7
    else:
        return category


# %%
def map_economic_equality_categories(category):
    if category == 'I dont know':
        return None  # What to do with this?
    elif category == 'differences in income should increase':
        return 1
    elif category == 'differences in income should decrease':
        return 5
    else:
        return category


# %%
def map_visits_categories(category):
    if category == '0 times':
        return 0
    elif category == '1 time' or category == 'one time':
        return 1
    elif category == '2 to 3 times':
        return 2.5
    elif category == '4 to 11 times':
        return 7.5
    elif category == '12 times or more':
        return 12
    else:
        return category




