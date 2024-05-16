# Thanks to the code of Julian Leeffers https://github.com/LISS-2024/j.m.q.leeffers/blob/main/Updated_code_Longitudinal_modelling.py
# TO DO:
# Drop IDs and cheating features
# df2 = df.drop(['nomem_encr', 'month', 'cs_m', 'cp014',
#                'cp015', 'cp016', 'cp017', 'cp018',
#                'cp038', 'cp068', 'cp076', 'cp080',
#                'cp081', 'cp147', 'cp148', 'cp149',
#                'cp152', 'cp154', 'cp165', 'cs001',
#                'cs002', 'cs070', 'cs283'], axis=1)
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.metrics import accuracy_score, confusion_matrix

from Preprocessing.PredictorEngineering import (
    PrefixToNumeric,
    NegaffectScale,
    PosAffectScale,
    map_optimism_categories,
    map_self_esteem_categories,
    map_BIG_V_categories,
    map_depressed_categories,
    map_health_categories,
    map_urban_categories,
    map_trust_others_categories,
    map_open_mind_categories,
    map_economic_equality_categories,
    map_visits_categories,
    map_education_categories,
)

# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

files = [
    "avars_201705_EN_1.0p.dta",
    "avars_201805_EN_1.0p.dta",
    "avars_201905_EN_1.0p.dta",
    "avars_202005_EN_1.0p.dta",
    "avars_202105_EN_1.0p.dta",
    "avars_202205_EN_1.0p.dta",  # May for each year.
    "ch17j_EN_1.0p.dta",
    "ch18k_EN_1.0p.dta",
    "ch19l_EN_1.0p.dta",
    "ch20m_EN_1.0p.dta",
    "ch21n_EN_1.0p.dta",
    "ch22o_EN_1.0p.dta",
    "cs17j_EN_1.0p.dta",
    "cs18k_EN_1.0p.dta",
    "cs19l_EN_1.0p.dta",
    "cs20m_EN_1.1p.dta",
    "cs21n_EN_1.1p.dta",
    "cs22o_EN_1.1p.dta",
    "cp17i_EN_1.0p.dta",
    "cp18j_EN_1.0p.dta",
    "cp19k_EN_1.0p.dta",
    "cp20l_EN_1.0p.dta",
    "cp21m_EN_1.0p.dta",
    "cp22n_EN_1.0p.dta",
    # 'cv17i_EN_1.0p.dta','cv18j_EN_1.0p.dta','cv19k_EN_1.0p.dta','cv20l_EN_1.0p.dta','cv22n_EN_1.0p.dta','cv23o_EN_1.0p.dta',
    # 'ci17j_EN_3.0p.dta','ci18k_EN_2.0p.dta','ci19l_EN_2.0p.dta','ci20m_EN_1.0p.dta',
]

dataframes = []

for file in files:
    if file[:2] == "av":  # Background Variables
        vars = {
            "nomem_encr": "nomem_encr",
            "Age": "leeftijd",
            # Measures age in numbers. Can also have a look at variable 'lftdcat', which groups age in categories.
            "Gender": "geslacht",  # Measures biological sex.
            "Pers_inc": "brutoink_f",  # Measures gross personal income.
            "Household_Income": "brutohh_f",  # Measures gross household income.
            "Urban": "sted",
            # Measures the urban character of place of residence. Seems like a 'semi'-geolocation indicator. Note: including leads to varying effects on performance of different models.
            "Education": "oplcat",  # Measures level of education in CBS (Statistics Netherlands) categories.
            "Origin": "herkomstgroep",
            # Measures (foreign) background. Dutch / Western / Non-Western, first gen / second gen. Note: missing code is '999'.
            #'Dwelling':'woning',                           # Measures the type of dwelling that the household inhabits.
            #'Domestic_sit' : 'woonvorm',                   # Measures the domestic situation of the household head.
            #'Occupation' :   #group occupations together into fewer categories.
        }
        single_wave_data = pd.io.stata.read_stata(absolute_path + file).loc[
            :, vars.values()
        ]
        single_wave_data.rename(
            columns={v: k for (k, v) in vars.items() if k != "nomem_encr"}, inplace=True
        )
        single_wave_data["Year"] = file[6:10]  # add corresponding year as string

    if file[:2] == "ch":  # Health survey (Core study)
        vars = {
            # Filename + number of the variable. I write it as such because it allows for automatically extracting the same column/variable over multiple years if needed.
            "nomem_encr": "nomem_encr",
            "Perc_health": file[:5]
            + "004",  # Measures person's own perception of their health, generally.
            "Depressed_gloomy": file[:5] + "014",
            "Nr_cigars": file[:5] + "132",
        }
        single_wave_data = pd.io.stata.read_stata(absolute_path + file).loc[
            :, vars.values()
        ]
        single_wave_data.rename(
            columns={v: k for (k, v) in vars.items() if k != "nomem_encr"}, inplace=True
        )
        single_wave_data["Year"] = "20" + file[2:4]  # add corresponding year as string

    if file[:2] == "cs":  # Leisure survey
        vars = {
            "nomem_encr": "nomem_encr",
            "Hum_rights_org": file[:5] + "028",
            "Environ_or_peace_org": file[:5] + "033",
            "Film_festivals": file[:5] + "099",
            "Museum_visits": file[:5] + "101",
            #'Sports_club' : file[:5] + '003',           # Measures whether someone has a connection to a sports club.
            #'Association' : file[:5] + '008',           # Measures whether someone has a connection to a cultural association or hobby club.
            #'Bus_organization' : file[:5] + '018',      # Measures whether someone has a connection to business, professional or agrarian organization.
        }
        single_wave_data = pd.io.stata.read_stata(absolute_path + file).loc[
            :, vars.values()
        ]
        single_wave_data.rename(
            columns={v: k for (k, v) in vars.items() if k != "nomem_encr"}, inplace=True
        )
        single_wave_data["Year"] = "20" + file[2:4]  # add year as string

    if file[:2] == "cp":  # Personality survey (Core study)
        vars = {
            "nomem_encr": "nomem_encr",
            "life_sat1": file[:5] + "014",
            "life_sat2": file[:5] + "015",
            "life_sat3": file[:5] + "016",
            "life_sat4": file[:5] + "017",
            "life_sat5": file[:5] + "018",
            "Trust_others": file[:5] + "019",
            "Open_mind": file[:5]
            + "103",  # Measures to what extent open-mindedness is important to someone.
            # BIG-V (Goldberg et al., 2006) #cp20l020 - cp20l069 See: https://ipip.ori.org/New_IPIP-50-item-scale.htm
            #     "extraV1": file[:5] + "020", "extraV2" : file[:5] + "025", "extraV3" : file[:5] + "030", "extraV4" : file[:5] + "035",  # Extraversion
            #     "agreeAb1" : file[:5] + "021", "agreeAb2" : file[:5] + "026", "agreeAb3": file[:5] + "031", "agreeAb4": file[:5] +"036",  # Agreeableness
            #     "conscious1" : file[:5] + "022", "conscious2" : file[:5] + "027", "conscious3" : file[:5] + "032", "conscious4" : file[:5] +"037",  # Conscientiousness
            #     "neurotic1" : file[:5] + "023", "neurotic2" : file[:5] + "028", "neurotic3" : file[:5] + "068", "neurotic4" : file[:5] + "038",  # Neuroticism
            #     "open1" : file[:5] + "024", "open2" : file[:5] + "029", "open3" : file[:5] +"034", "open4" : file[:5] +"039",   # Openness
            "BIG-V_1_": file[:5] + "020",
            "BIG-V_2_": file[:5] + "021",
            "BIG-V_3_": file[:5] + "022",
            "BIG-V_4_": file[:5] + "023",
            "BIG-V_5_": file[:5] + "024",
            "BIG-V_6_": file[:5] + "025",
            "BIG-V_7_": file[:5] + "026",
            "BIG-V_8_": file[:5] + "027",
            "BIG-V_9_": file[:5] + "028",
            "BIG-V_10_": file[:5] + "029",
            "BIG-V_11_": file[:5] + "030",
            "BIG-V_12_": file[:5] + "031",
            "BIG-V_13_": file[:5] + "032",
            "BIG-V_14_": file[:5] + "033",
            "BIG-V_15_": file[:5] + "034",
            "BIG-V_16_": file[:5] + "035",
            "BIG-V_17_": file[:5] + "036",
            "BIG-V_18_": file[:5] + "037",
            "BIG-V_19_": file[:5] + "038",
            "BIG-V_20_": file[:5] + "039",
            "BIG-V_21_": file[:5] + "040",
            "BIG-V_22_": file[:5] + "041",
            "BIG-V_23_": file[:5] + "042",
            "BIG-V_24_": file[:5] + "043",
            "BIG-V_25_": file[:5] + "044",
            "BIG-V_26_": file[:5] + "045",
            "BIG-V_27_": file[:5] + "046",
            "BIG-V_28_": file[:5] + "047",
            "BIG-V_29_": file[:5] + "048",
            "BIG-V_30_": file[:5] + "049",
            "BIG-V_31_": file[:5] + "050",
            "BIG-V_32_": file[:5] + "051",
            "BIG-V_33_": file[:5] + "052",
            "BIG-V_34_": file[:5] + "053",
            "BIG-V_35_": file[:5] + "054",
            "BIG-V_36_": file[:5] + "055",
            "BIG-V_37_": file[:5] + "056",
            "BIG-V_38_": file[:5] + "057",
            "BIG-V_39_": file[:5] + "058",
            "BIG-V_40_": file[:5] + "059",
            "BIG-V_41_": file[:5] + "060",
            "BIG-V_42_": file[:5] + "061",
            "BIG-V_43_": file[:5] + "062",
            "BIG-V_44_": file[:5] + "063",
            "BIG-V_45_": file[:5] + "064",
            "BIG-V_46_": file[:5] + "065",
            "BIG-V_47_": file[:5] + "066",
            "BIG-V_48_": file[:5] + "067",
            "BIG-V_49_": file[:5] + "068",
            "BIG-V_50_": file[:5] + "069",
            # Self-esteem, Rosenberg (2015): cp20l070 - cp20l079 See: https://fetzer.org/sites/default/files/images/stories/pdf/selfmeasures/Self_Measures_for_Self-Esteem_ROSENBERG_SELF-ESTEEM.pdf
            "Self-esteem_1_": file[:5] + "070",
            "Self-esteem_2_": file[:5] + "071",
            "Self-esteem_3_": file[:5] + "072",
            "Self-esteem_4_": file[:5] + "073",
            "Self-esteem_5_": file[:5] + "074",
            "Self-esteem_6_": file[:5] + "075",
            "Self-esteem_7_": file[:5] + "076",
            "Self-esteem_8_": file[:5] + "077",
            "Self-esteem_9_": file[:5] + "078",
            "Self-esteem_10_": file[:5] + "079",
            # Inverse scored: 79, 74, 78, 77, 72 ; 10,5,9,8,3
            # PANAS (positive and negative affect scale (= cp23o146 - cp23o165: ) Watson, D., Clark, L. A., & Tellegen, A. (1988). Development and validation of brief measures of positive and negative affect:
            # The PANAS scales. Journal of Personality and Social Psychology, 54, 1063-1070. https://doi.org/10.1037/0022-3514.54.6.1063)
            # PANAS Positive affect
            "pos_1": file[:5] + "146",
            "pos_2": file[:5] + "148",
            "pos_3": file[:5] + "150",
            "pos_4": file[:5] + "154",
            "pos_5": file[:5] + "155",
            "pos_6": file[:5] + "157",
            "pos_7": file[:5] + "159",
            "pos_8": file[:5] + "161",
            "pos_9": file[:5] + "162",
            "pos_10": file[:5] + "164",
            # PANAS Negative affect
            "neg_1": file[:5] + "147",
            "neg_2": file[:5] + "149",
            "neg_3": file[:5] + "151",
            "neg_4": file[:5] + "152",
            "neg_5": file[:5] + "153",
            "neg_6": file[:5] + "156",
            "neg_7": file[:5] + "158",
            "neg_8": file[:5] + "160",
            "neg_9": file[:5] + "163",
            "neg_10": file[:5] + "165",
            # Optimism (Pessimism inverse scored) (Scheier, Carver, and Bridges, 1994): cp20l198 - cp20l207 See: https://www.cmu.edu/dietrich/psychology/pdf/scales/LOTR_Scale.pdf
            "Optimism_1_": file[:5] + "198",
            "Optimism_2_": file[:5] + "199",
            "Optimism_3_": file[:5] + "200",
            "Optimism_4_": file[:5] + "201",
            "Optimism_5_": file[:5] + "202",
            "Optimism_6_": file[:5] + "203",
            "Optimism_7_": file[:5] + "204",
            "Optimism_8_": file[:5] + "205",
            "Optimism_9_": file[:5] + "206",
            "Optimism_10_": file[:5] + "207",
            "Date": file[:5] + "191",
            # Inverse scored: 3,7,9
        }
        single_wave_data = pd.io.stata.read_stata(absolute_path + file).loc[
            :, vars.values()
        ]
        single_wave_data.rename(
            columns={v: k for (k, v) in vars.items() if k != "nomem_encr"}, inplace=True
        )
        single_wave_data["Year"] = "20" + file[2:4]  # add year as string

        # Create the feature engineered target variable SWLS
        # single_wave_data = IDtoInt(single_wave_data)
        single_wave_data = PrefixToNumeric(single_wave_data)
        # single_wave_data = CalculateLifeSatScore(single_wave_data)

        # calculate feature engineered predictor variable
        single_wave_data = NegaffectScale(single_wave_data)
        single_wave_data = PosAffectScale(single_wave_data)

        # Target variable back to categorical
        # single_wave_data = binning(single_wave_data)

    if file[:2] == "ci":  # Economic Situation (Income) survey (Core study)
        vars = {
            # Filename + number of the variable. I write it as such because it allows for automatically extracting the same column/variable over multiple years if needed.
            "nomem_encr": "nomem_encr",
            "Current_fin_sit": file[:5]
            + "006",  # Measures person's satisfaction with their financial situation.
            "Current_eco_sit": file[:5] + "007",
            # Measures person's satisfaction with the current economic situation in the Netherlands.
        }
        single_wave_data = pd.io.stata.read_stata(absolute_path + file).loc[
            :, vars.values()
        ]
        single_wave_data.rename(
            columns={v: k for (k, v) in vars.items() if k != "nomem_encr"}, inplace=True
        )

        single_wave_data["Year"] = "20" + file[2:4]  # add year as string

    dataframes.append(single_wave_data)

import pandas as pd

# Concatenate the dataframes
stacked_df = pd.concat(dataframes, axis=0)

# Convert 'nomem_encr' to integer type
stacked_df["nomem_encr"] = stacked_df["nomem_encr"].astype(int)

# Set 'nomem_encr' and 'Year' as indices
stacked_df.set_index(["nomem_encr", "Year"], inplace=True)

# Sort the dataframe based on indices
stacked_df.sort_index(level=["nomem_encr", "Year"], inplace=True)

# Select numeric and non-numeric columns
numeric_cols = stacked_df.select_dtypes(include=["number"])
non_numeric_cols = stacked_df.select_dtypes(exclude=["number"])

# Aggregate numeric columns
merged_numeric_df = numeric_cols.groupby(level=["nomem_encr", "Year"]).sum()

# Aggregate non-numeric columns
merged_non_numeric_df = non_numeric_cols.groupby(level=["nomem_encr", "Year"]).first()

# Merge numeric and non-numeric DataFrames
merged_stacked_df = pd.merge(
    merged_numeric_df, merged_non_numeric_df, left_index=True, right_index=True
)

# Save the merged dataframe to CSV
merged_stacked_df.to_csv(absolute_path + "Longitudinal_dataset.csv")

# Display the merged dataframe
merged_stacked_df.iloc[:20, :]

# %%
#### Start here. Loading the dataset above takes ~5-6 minutes
# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

merged_df = pd.read_csv(absolute_path + "Longitudinal_dataset.csv", low_memory=False)
merged_df = merged_df.set_index(["nomem_encr", "Year"])


def CalculateLifeSatScore(df):
    # Define the columns to check for life satisfaction
    life_sat_columns = ["life_sat1", "life_sat2", "life_sat3", "life_sat4", "life_sat5"]
    # Initialize the 'LS_SCORE_' column with the sum of the life satisfaction columns
    df["LS_SCORE_"] = df[life_sat_columns].sum(axis=1)
    # Check for any zeros in the life satisfaction columns and set 'LS_SCORE_' to 0 if found
    df.loc[df[life_sat_columns].eq(0).any(axis=1), "LS_SCORE_"] = 0
    return df


merged_df = CalculateLifeSatScore(merged_df)
print(merged_df.LS_SCORE_.value_counts())


# %%
def remove_unknown_targets(response):
    if response in [0, 0.0, -9, "0", "0.0", "-9", "I don't know"]:
        response = None
    return response


for col in merged_df.columns:
    if "LS_SCORE_" in col:
        merged_df[col] = merged_df[col].apply(remove_unknown_targets)
print(merged_df.LS_SCORE_.value_counts())

# merged_df["LS_SCORE"] = merged_df["LS_SCORE_"].astype(int)
merged_df["LS_SCORE_"].dropna(inplace=True)

# %%
# Define the bins edges based on the SWLS score
# Adjust the bins to align with the distribution of LS_SCORE_ values
bins = [4, 19, 24, 35]  # Example bin edges that you might adjust

# Define the labels for the bins
# labels = ["Unsatisfied", "Neutral", "Satisfied"]
labels = [1, 2, 3]

# %%
# Create a new column 'Satisfaction_Level' with the binned values
merged_df["Satisfaction_level"] = pd.cut(
    merged_df["LS_SCORE_"], bins=bins, labels=labels, include_lowest=True
)

# Check the distribution of the new Satisfaction Levels
print(merged_df["Satisfaction_level"].value_counts())
# %%
# Reset the index to have 'nomem_encr' and 'Year' as regular columns
merged_df.reset_index(inplace=True)

# Create a boolean mask for rows with 'Year' == 2022 and 'Satisfaction_level' == None.
mask = (merged_df["Year"] == 2022) & (merged_df["Satisfaction_level"].isna())

# Filter the DataFrame to keep only rows that don't match the mask. Removing rows where Year 2022 and Vote == None
merged_df = merged_df[~mask]

# Set 'nomem_encr' and 'Year' as indices again
merged_df.set_index(["nomem_encr", "Year"], inplace=True)

# %%
# If we do not load the data from the 2023 wave, this cell can be left out. If we want to fill NAs from 2022 with votes reported in 2023, we need to change this cell.

# Only keep the subjects who have available data across all years from the last election year until the last year before the election year of interest + the year containing the target (so 2017,2018,2019,2020,2022):

# Group by 'nomem_encr' and check if the specified 'Year' values are present
selected_nomem_encr = merged_df.groupby("nomem_encr")["Year"].apply(
    lambda x: set(x) >= {2017, 2018, 2019, 2020, 2022}
)

# Filter the DataFrame to keep only the rows of selected 'nomem_encr' values
filtered_df = merged_df[
    merged_df.index.get_level_values("nomem_encr").isin(
        selected_nomem_encr[selected_nomem_encr].index
    )
]

# %%
# setting of index can be left out here
filtered_df = filtered_df.set_index("Year", append=True)
merged_df = filtered_df

# %%
# Store the original scores for error analysis later on.
original_SWL_scores = merged_df.Satisfaction_level.dropna()
original_SWL_scores

# Apply the function to columns containing 'Education'
for col in merged_df.columns:
    if "Education" in col:
        merged_df[col] = merged_df[col].apply(map_education_categories)

# Apply the function to columns containing 'Urban'
for col in merged_df.columns:
    if "Urban" in col:
        merged_df[col] = merged_df[col].apply(map_urban_categories)

# Apply the function to columns containing 'health'
for col in merged_df.columns:
    if "health" in col:
        merged_df[col] = merged_df[col].apply(map_health_categories)


# Apply the function to columns containing 'Film_festivals'
for col in merged_df.columns:
    if "Film_festivals" in col or "Museum_visits" in col:
        merged_df[col] = merged_df[col].apply(map_visits_categories)

# Apply the function to columns containing 'Trust_others'
for col in merged_df.columns:
    if "Trust_others" in col:
        merged_df[col] = merged_df[col].apply(map_trust_others_categories)

# Apply the function to columns containing 'Open_mind'
for col in merged_df.columns:
    if "Open_mind" in col:
        merged_df[col] = merged_df[col].apply(map_open_mind_categories)

# Apply the function to columns containing 'BIG-V_1_'
for col in merged_df.columns:
    if "BIG-V_" in col:
        merged_df[col] = merged_df[col].apply(map_BIG_V_categories)

# Apply the function to columns containing 'Self-esteem_'
for col in merged_df.columns:
    if "Self-esteem_" in col:
        merged_df[col] = merged_df[col].apply(map_self_esteem_categories)

# Apply the function to columns containing 'Optimism_'
for col in merged_df.columns:
    if "Optimism_" in col:
        merged_df[col] = merged_df[col].apply(map_optimism_categories)

# Apply the function to columns containing 'Economic_equality'
for col in merged_df.columns:
    if "Economic_equality" in col:
        merged_df[col] = merged_df[col].apply(map_economic_equality_categories)

# Apply the function to columns containing 'Depressed'
for col in merged_df.columns:
    if "Depressed" in col:
        merged_df[col] = merged_df[col].apply(map_depressed_categories)
# %%
# Checkpoint
merged_df.to_csv(absolute_path + "Checkpoint.csv")

# %%
import pandas as pd

# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

merged_df = pd.read_csv(absolute_path + "Checkpoint.csv", low_memory=False)
merged_df = merged_df.set_index("nomem_encr")
# print(merged_df.shape)

# %%
"""# Optional code to leave out columns.
left_out_columns = [col for col in merged_df.columns if col in [
    'Left_Right_Scale',
    ]]

merged_df = merged_df.drop(columns=left_out_columns)"""

# %%
"""# Identify indices of subjects with household income greater than 30000. There are still people with a consistent household income around 20000 - 30000. But beyond that these are unlikely values.
indices_high_income = merged_df.index[merged_df['Household_Income'] > 30000]

# Create a DataFrame containing only subjects with high income and their income column
high_income_df = merged_df.loc[indices_high_income, ['Household_Income']]

# Display the DataFrame
high_income_df[:10]

# Note that rows with the Year 2022 are still included here which explains the 0.0 values. These rows are dropped later."""

# %%
left_out_columns = [
    col
    for col in merged_df.columns
    if col
    in [
        # Task to do: make up your mind, using the literature: which variables to include exclude at first.
        # "Pers_inc",
        #'Gender',
        "Hum_rights_org",
        "Environ_or_peace_org",
        "Museum_visits",
        "Film_festivals",  # Task to do: Combine cultural items into a single column/feature
        # "Depressed_gloomy",
        "Trust_others",
        "Open_mind",  # Open_mindedness column, or Personality variables generally, have some NaN's
        "Feel_Dutch",
        "Nr_cigars",
    ]
]

# merged_df = merged_df.drop(columns=left_out_columns)
# Note that rows with the Year 2022 are still included here which explains the 0.0 values. These rows are dropped later.
# %%
# Replace household income values higher than 30000 with None.
merged_df.loc[merged_df["Household_Income"] > 30000, "Household_Income"] = None

# %%
# Turning numeric columns into numeric datatype.
# %%
categorical_cols = ["Env", "Hum", "Gen", "Ori", "Urb"]
for col in merged_df.columns:
    if col[:3] not in categorical_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="raise")

# %%
# Splitting targets and predictors before imputing values.
targets = merged_df[merged_df["Year"] == 2022][
    ["Year", "Satisfaction_level"]
].reset_index()
predictors = merged_df[merged_df["Year"] != 2022].drop(columns="Satisfaction_level")


# %%
predictors.reset_index(inplace=True)
predictors.set_index(["nomem_encr", "Year"], inplace=True)

# WE will now continue with creating features
# %%
# CREATE dummy variables
predictors = pd.get_dummies(predictors, drop_first=False)

# %% [markdown]
# ### Imputation

# %%
# Interpolate within each subject group where possible
predictors = predictors.groupby("nomem_encr", group_keys=False).apply(
    lambda group: group.interpolate(method="linear")
)

# Fill forward within each subject group where possible
predictors = predictors.groupby("nomem_encr", group_keys=False).apply(
    lambda group: group.ffill()
)

# Fill backward within each subject group where possible
predictors = predictors.groupby("nomem_encr", group_keys=False).apply(
    lambda group: group.bfill()
)


# %%
# Dropped where more than 5% NaN values are present and impute remaining NaN values using Multiple Imputation
# This dataframe to compare different imputation methods
merged_df.to_csv(dir_path + "CleanData/BEFORE_Imputation.csv")

# Step 3: Perform Multiple Imputation on the remaining rows
# Instantiate the IterativeImputer
from sklearn.impute import IterativeImputer

# Step 1: Identify indices of rows with more than 5% NaN values
nan_percentage_per_row = predictors.isna().mean(axis=1) * 100
rows_to_drop = nan_percentage_per_row[nan_percentage_per_row > 5].index

# Step 2: Drop all rows for indices with more than 5% NaN values
predictors_cleaned = predictors.drop(rows_to_drop)


imputer = IterativeImputer(max_iter=5, random_state=42)

# Fit and transform the imputer on the remaining data
predictors_imputed = pd.DataFrame(
    imputer.fit_transform(predictors_cleaned),
    columns=predictors_cleaned.columns,
    index=predictors_cleaned.index,
)

# %%
# Checkpoint2
predictors_imputed.to_csv(absolute_path + "Checkpoint2.csv")

# %%
import pandas as pd

# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

predictors_imputed = pd.read_csv(absolute_path + "Checkpoint2.csv", low_memory=False)

# %%
# Adding targets again
merged_df = pd.concat([predictors_imputed, targets], axis=0)


# %%
# Set the indices again
merged_df.set_index(["nomem_encr", "Year"], inplace=True)
merged_df.sort_index(level=["nomem_encr", "Year"], inplace=True)


# %%
# Combine predictors and the targets in the same rows
merged_df = merged_df.groupby(level=0, group_keys=False).apply(
    lambda group: group.bfill()
)  # Fill in the target value into the earlier years before taking out rows with Year 2022

# Take out the rows with Year 2022 since we will not be predicting with these
merged_df = merged_df.reset_index("Year")
merged_df = merged_df[merged_df["Year"] != 2022]
print(merged_df.Satisfaction_level)

# %%
# from sklearn.preprocessing import OrdinalEncoder
#
# # Encode the target variable using LabelEncoder
# ordinal_encoder = OrdinalEncoder()
# merged_df["Satisfaction_level"] = ordinal_encoder.fit_transform(
#     merged_df["Satisfaction_level"]
# )
#
# # Add 1 to the encoded values. This is needed for RF++, because it requires target variables to be in the range[1, classes]
# merged_df["Satisfaction_level"] += 1
# print(merged_df["Satisfaction_level"])
# %% [markdown]
# ### Feature engineering


# %%
def reverse_score_5_likert(score):
    score = (score * -1) + 6
    return score


# %%
# Summing up self-esteem columns.
def reverse_score_7_likert(
    score,
):  # Check what happens to NaN values? Can this function be used for Immigration variables in the IM_ code block above?
    score = (score * -1) + 8
    return score


for col in merged_df.columns:
    if "Self" in col:
        if col[11:14] in ["_3_", "_5_", "_8_", "_9_", "_10"]:
            merged_df[col] = merged_df[col].apply(reverse_score_7_likert)

# Sum up the scores for Self-esteem columns.
merged_df["Esteem"] = merged_df[
    [col for col in merged_df.columns if "esteem" in col]
].sum(axis=1)

# %%
# Summing up Optimism columns.
for col in merged_df.columns:
    if "Optimism" in col:
        if col[8:11] in ["_3_", "_7_", "_9_"]:
            merged_df[col] = merged_df[col].apply(reverse_score_5_likert)

# Sum up the scores for Optimism columns.
merged_df["Optim"] = merged_df[
    [col for col in merged_df.columns if "Optimism" in col]
].sum(axis=1)

# %%
# Reverse scores for inverted scales. BIG-V 50-item Sample Questionnaire can be found here: https://ipip.ori.org/New_IPIP-50-item-scale.htm
count = 0
for col in merged_df.columns:  # Can use enumerate instead.
    if "BIG" in col:
        count += 1
        if count % 2 == 0:
            merged_df[col] = merged_df[col].apply(reverse_score_5_likert)

# Sum up the scores for BIG-V personality traits.
merged_df["Extraversion"] = merged_df[
    [
        "BIG-V_1_",
        "BIG-V_6_",
        "BIG-V_11_",
        "BIG-V_16_",
        "BIG-V_21_",
        "BIG-V_26_",
        "BIG-V_31_",
        "BIG-V_36_",
        "BIG-V_41_",
        "BIG-V_46_",
    ]
].sum(axis=1)
merged_df["Agreeableness"] = merged_df[
    [
        "BIG-V_2_",
        "BIG-V_7_",
        "BIG-V_12_",
        "BIG-V_17_",
        "BIG-V_22_",
        "BIG-V_27_",
        "BIG-V_32_",
        "BIG-V_37_",
        "BIG-V_42_",
        "BIG-V_47_",
    ]
].sum(axis=1)
merged_df["Conscientiousness"] = merged_df[
    [
        "BIG-V_3_",
        "BIG-V_8_",
        "BIG-V_13_",
        "BIG-V_18_",
        "BIG-V_23_",
        "BIG-V_28_",
        "BIG-V_33_",
        "BIG-V_38_",
        "BIG-V_43_",
        "BIG-V_48_",
    ]
].sum(axis=1)
merged_df["Emotional_Stability"] = merged_df[
    [
        "BIG-V_4_",
        "BIG-V_9_",
        "BIG-V_14_",
        "BIG-V_19_",
        "BIG-V_24_",
        "BIG-V_29_",
        "BIG-V_34_",
        "BIG-V_39_",
        "BIG-V_44_",
        "BIG-V_49_",
    ]
].sum(axis=1)
merged_df["Intellect_Imagination"] = merged_df[
    [
        "BIG-V_5_",
        "BIG-V_10_",
        "BIG-V_15_",
        "BIG-V_20_",
        "BIG-V_25_",
        "BIG-V_30_",
        "BIG-V_35_",
        "BIG-V_40_",
        "BIG-V_45_",
        "BIG-V_50_",
    ]
].sum(axis=1)

# Drop the individual Personality survey questions.
merged_df = merged_df.drop(
    columns=[
        col
        for col in merged_df.columns
        if "BIG" in col
        or "Self" in col
        and col != "Self_Esteem"
        or "Optimism" in col
        and col != "Optimism"
    ]
)


# %%
import matplotlib.pyplot as plt
import matplotlib

# # Setting font to Times New Roman
matplotlib.rcParams["font.family"] = "Times New Roman"

# Checking the class (im)balance in the target variable
# print(merged_df.LS_SCORE_.value_counts())
# class_counts = merged_df["LS_SCORE_"].value_counts()

# # Checking the class (im)balance in the alternative target variable
# print(merged_df["LS_SCORE_"].value_counts())
class_counts = merged_df["Satisfaction_level"].value_counts()

# # Plotting
# plt.figure(figsize=(8, 6))
# bars = class_counts.plot(kind="bar")
# plt.title("Class Balance in Satisfaction With Life Scale (2022)")
# plt.xlabel("Life Satisfaction")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
#
# # Add values on top of the bars
# for bar, count in zip(bars.patches, class_counts):
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + 0.5,
#         f"{count}",
#         ha="center",
#         va="bottom",
#     )
#
# plt.show()

# %% [markdown]
# The following is an attempt at longitudinal modelling.


# %%
# Checkpoint3
merged_df.to_csv(absolute_path + "Checkpoint3.csv")
# %%
import pandas as pd

# Personal and absolute path to access the datafiles.
dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
absolute_path = dir_path + "RawData/"

merged_df = pd.read_csv(absolute_path + "Checkpoint3.csv", low_memory=False)
RF_plus_plus_df = merged_df.set_index(
    "nomem_encr"
)  # RF++ only requires the unique IDs as index
print(RF_plus_plus_df["Satisfaction_level"].value_counts())

# %%
# Putting the target as the last column of the dataframe, as this is also required for running RF++

# Get the list of columns excluding the target column
columns = [col for col in RF_plus_plus_df.columns if col != "Satisfaction_level"]

# Add the target column at the end of the list
columns.append("Satisfaction_level")

# Reindex the DataFrame with the new order of columns
RF_plus_plus_df = RF_plus_plus_df.reindex(columns=columns)


# %%
# Directly using train_test_split on the longitudinal dataset will cause rows pertaining to the same individuals to be divided across the train and test sets.
# The following code overcomes this problem.
RF_plus_plus_df.to_csv(absolute_path + "RF_plus_plus.csv")

# Aggregate the target column to 1 row per ID/nomem_encr.
aggregated_data = RF_plus_plus_df.groupby(["nomem_encr"]).agg(
    {"Satisfaction_level": "last"}
)
# %%
len(aggregated_data)
# %%
# import pandas as pd
#
# # Personal and absolute path to access the datafiles.
# dir_path = "C:/Users/fenna/PycharmProjects/fennaz/Data/"
# absolute_path = dir_path + "RawData/"
#
# RF_plus_plus_df = pd.read_csv(absolute_path + "RF_plus_plus.csv", low_memory=False)
# RF_plus_plus_df = RF_plus_plus_df[RF_plus_plus_df["Year"] != 2022]
# RF_plus_plus_df = RF_plus_plus_df.set_index(["nomem_encr", "Year"])

print(aggregated_data["Satisfaction_level"].info())
print(aggregated_data["Satisfaction_level"].value_counts())
# %%
# The following code was used to obtain the proportions of voters in the same for bins, for ceating Table 1 in the thesis document.
# Note that the step in which the parties are grouped into bins should be left out when running this code. Also, label encoding of the target column should be left out when doing so.
#
# sample_percentages_per_score = (
#     round(aggregated_data.value_counts() / aggregated_data.value_counts().sum(), 3)
#     * 100
# )
# for i, j in enumerate(sample_percentages_per_score):
#     print(str(round(j, 1)) + "% " + str(sample_percentages_per_score.index[i]))


# %%
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)

# Use stratified train_test_split to create stratified train and test sets based on unique IDs.
train_group, test_group = train_test_split(
    aggregated_data,
    test_size=0.2,
    random_state=42,
    stratify=aggregated_data["Satisfaction_level"],
)

# Take the indices from these sets as the indices by which to subsequently split the complete dataset.
train_ids = list(train_group.index)
test_ids = list(test_group.index)

train_df = RF_plus_plus_df.loc[train_ids].reset_index()
test_df = RF_plus_plus_df.loc[test_ids].reset_index()

# %%
# Parameters for running RF++ on test set
print("Rows in training + validation set:", train_df.shape[0])
print("Rows in test set:", test_df.shape[0])
print(
    "Amount of variables (excluding nomem_encr and the target):", train_df.shape[1] - 2
)

# %%
# Save the train and test set as a text documents without file extensions for RF++
train_file_path = "C:/Users/fenna/RF_files/RF++Train"
test_file_path = "C:/Users/fenna/RF_files/RF++Test"

# Save the train DataFrame to a text document without column names
train_df.to_csv(train_file_path + ".txt", sep="\t", index=False, header=False)

# Save the test DataFrame to a text document without column names
test_df.to_csv(test_file_path + ".txt", sep="\t", index=False, header=False)

# %%
# Save the train and test sets as text documents without file extensions for HistoricalRF
train_file_path = "C:/Users/fenna/RF_files/Hist_RF_train"
test_file_path = "C:/Users/fenna/RF_files/Hist_RF_test"

# Save the train DataFrame to a text document without column names
train_df.to_csv(train_file_path + ".txt", sep="\t", index=False)

# Save the test DataFrame to a text document without column names
test_df.to_csv(test_file_path + ".txt", sep="\t", index=False)


# %% Needs to be stratified for imbalanced classes
# Use stratified train_test_split to create stratified train and validation sets for hyper-parameter tuning.
cross_val_train_group, cross_val_validation_group = train_test_split(
    train_group,
    test_size=0.25,
    random_state=42,
    stratify=train_group["Satisfaction_level"],
)

cross_val_train_ids = list(cross_val_train_group.index)
cross_val_validation_ids = list(cross_val_validation_group.index)

cv_train_df = RF_plus_plus_df.loc[cross_val_train_ids].reset_index()
cv_validation_df = RF_plus_plus_df.loc[cross_val_validation_ids].reset_index()

# %%
# Parameters for running RF++ on validation set
print("Rows in training set:", cv_train_df.shape[0])
print("Rows in validation set:", cv_validation_df.shape[0])
print(
    "Amount of variables (excluding nomem_encr and the target):", train_df.shape[1] - 2
)

# %%
# Save the cv_train and cv_validation set as a text documents without file extensions for RF++
cv_train_file_path = "C:/Users/fenna/RF_files/manual/Hist_RF_train"
cv_test_file_path = "C:/Users/fenna/RF_files/Hist_RF_train/manual/RF++CV_Validation"

# Save the cv_train DataFrame to a text document without column names
cv_train_df.to_csv(cv_train_file_path + ".txt", sep="\t", index=False, header=False)

# Save the cv_validation DataFrame to a text document without column names
cv_validation_df.to_csv(cv_test_file_path + ".txt", sep="\t", index=False, header=False)

# %%
# Save the cv_train and cv_validation set as a text documents without file extensions for Historical_RF
cv_train_file_path = "C:/Users/fenna/RF_files/Hist_RF_CV_Train"
cv_test_file_path = "C:/Users/fenna/RF_files/Hist_RF_CV_Validation"

# Save the cv_train DataFrame to a text document with column names
cv_train_df.to_csv(cv_train_file_path + ".txt", sep="\t", index=False)

# Save the cv_validation DataFrame to a text document without column names
cv_validation_df.to_csv(cv_test_file_path + ".txt", sep="\t", index=False)

# %%
# This can be used to identify the feature importance columns from RF++ (the report only states the index of the column).
train_df.iloc[:, 8].name
# %%
# Set the multi-index
RF_df = RF_plus_plus_df.reset_index()
RF_df.set_index(["nomem_encr", "Year"], inplace=True)

# %%
import numpy as np
import seaborn as sns

# # Dummy vs RF_last vs RF_mean

# Initialize dictionaries to store performances
performance_cv = {}
performance_test = {}

for measurement in ["last", "mean"]:
    # Aggregate the data depending on the measurement used
    columns = {col: measurement for col in RF_df.columns}
    columns["Satisfaction_level"] = "last"
    aggregated_data = RF_df.groupby(["nomem_encr"]).agg(
        columns
    )  # Put a # in front of .groupby to make it run RF on all rows.

    # Create train and test sets using the unique IDs from the stratified split.
    train_set = aggregated_data.loc[train_ids]
    test_set = aggregated_data.loc[test_ids]
    X_train = train_set.drop(columns="Satisfaction_level")
    y_train = train_set["Satisfaction_level"]
    X_test = test_set.drop(columns="Satisfaction_level")
    y_test = test_set["Satisfaction_level"]

    """    X_train_val_set # TO DO: Still need to replace cross-validation
    y_train_val_set
    X_validation_set
    y_validation_set"""

    # Initialize RandomForest model
    RF_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize empty list to store feature importances across folds
    feature_importances_across_folds = []

    # Iterate over each fold
    for train_index, _ in cv.split(X_train):
        # Train the model
        RF_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        # Extract feature importance from the trained model
        feature_importances_across_folds.append(RF_model.feature_importances_)

    # Calculate average feature importances across folds
    average_feature_importances = np.mean(feature_importances_across_folds, axis=0)

    # Create DataFrame to store feature names and their corresponding importance
    feature_importance_df_predictors = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": average_feature_importances}
    )
    # Sort the DataFrame by importance in descending order
    feature_importance_df_predictors = feature_importance_df_predictors.sort_values(
        by="Importance", ascending=False
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df_predictors["Feature"],
        feature_importance_df_predictors["Importance"],
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Average Feature Importance for the RF model with measurement: " + measurement
    )
    plt.show()

    # Define a function for model training and cross-validation
    def train_and_cross_validate_model(model, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="accuracy"
        )  # adjust to sensitivity /recall and/or Auroc
        return scores

    # Initialize models including additional ones
    classifiers = {
        "Dummy Classifier": DummyClassifier(
            strategy="stratified"
        ),  # Baseline, applicale for imbalanced classes
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Train and cross-validate each model
    for name, model in classifiers.items():
        scores = train_and_cross_validate_model(model, X_train, y_train)
        if "Dummy" not in name:
            performance_cv[name + " using " + measurement] = scores
        else:
            performance_cv[name] = scores

    # Test each model on the hold-out test set
    confusion_matrices = {}
    misclassified_rows = {}  # Initialize dictionary to store misclassified rows

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(
            y_test, y_pred
        )  # also adjust to ensitivity /recall and/or Auroc
        if "Dummy" not in name:
            performance_test[name + " using " + measurement] = accuracy
        else:
            performance_test[name] = accuracy

        confusion_matrices[name + " using " + measurement] = confusion_matrix(
            y_test, y_pred
        )

        # Find misclassified rows
        misclassified_indices = np.where(y_pred != y_test)[0]
        misclassified_data = X_test.iloc[misclassified_indices]

        # Store misclassified rows in the dictionary
        misclassified_rows[name + " using " + measurement] = misclassified_data

    # Plot confusion matrices
    for name, cm in confusion_matrices.items():
        if "Dummy" not in name:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5, square=True
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(f"Confusion Matrix for {name}")
            plt.show()

    # Convert the dictionary of misclassified rows to DataFrame
    df_misclassified = pd.concat(misclassified_rows, names=["Model"])

    # Show the DataFrame containing misclassified rows
    df_misclassified

print("\nPerformance across validation sets:")
for name, acc in performance_cv.items():
    print(f" Model: {name}, Accuracy: {round(np.mean(acc), 3)}")

# Print performance on the hold-out test set
print("\nPerformance on hold-out test set:")
for name, acc in performance_test.items():
    print(f" Model: {name}, Accuracy: {round(acc, 3)}")

# Convert the dictionary of scores to DataFrame
df_test = pd.DataFrame(performance_test, index=["Accuracy on hold-out test set"])

# Plot model performances across test sets
plt.figure(figsize=(10, 6))
df_test.loc["Accuracy on hold-out test set"].sort_values().plot(
    kind="barh", color="skyblue"
)
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.title("Model Performances on Hold-out Test Set")
plt.grid(axis="x")
plt.show()

# TODO add functions or revise this function to test various imputation methods and class imbalance handlings
# %% [markdown]
# #### The additional majority vote step for running an unmodified RF on all observations:

# %%
"""# Subject-level accuracy computation. Adjust the code above to make RF run on all observations before running this.

predictions = [y_pred]  # array of binary predictions.

# Reshape the array into groups of 4.
predictions_reshaped = np.reshape(predictions, (-1, 4))

# Calculate the mode along the rows.
modes = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions_reshaped)

# Get the last number from each group
last_numbers = predictions_reshaped[:, -1]

# Replace the mode with the last number if there's a tie.
for i in range(len(modes)):
    counts = np.bincount(predictions_reshaped[i])
    if counts[0] == counts[1]:  #  Tie
        modes[i] = last_numbers[i] # Replace

subject_level_targets = np.array(y_test)[::4] # Take every 4th value in order to get the right array-length.

round(accuracy_score(subject_level_targets, modes),4)"""
