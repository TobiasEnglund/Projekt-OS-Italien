import dash
from dash import Dash, html, dcc, callback, Output, Input
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Datasets

athlete_events = pd.read_csv("./Data/athlete_events.csv")
noc_regions = pd.read_csv("./Data/noc_regions.csv")
best_italian_years = pd.read_excel("./Data/best_italian_years.xlsx")

# Dataframes

medals = athlete_events[athlete_events['Medal'].isin(['Gold', 'Silver', 'Bronze'])]
medals_count = medals.groupby(['Sport', 'Team', 'Medal']).size().reset_index(name='Count')

url = 'https://www.worldometers.info/world-population/population-by-country/'
tables = pd.read_html(url)
df = tables[0]

merged_df = pd.merge(athlete_events, noc_regions, on='NOC', how='left')
merged_df = pd.merge(merged_df, df[['Country (or dependency)', 'Population  (2023)']], left_on='region', right_on='Country (or dependency)', how='left')
merged_df.dropna(subset=['Population  (2023)'], inplace=True)

medals_per_capita = merged_df.groupby('region')['Medal'].count() / merged_df.groupby('region')['Population  (2023)'].mean()

top_countries = medals_per_capita.sort_values(ascending=False)[:10].reset_index()
top_countries['Population'] = merged_df.groupby('region')['Population  (2023)'].mean().loc[top_countries['region']].values
top_countries.rename(columns={0: 'Medals'}, inplace=True)

italy_medals = medals.query("Team == 'Italy'")
italy_medals.value_counts("Medal")
italy_medals_count = italy_medals.value_counts("Sport")

medals_per_game = italy_medals.groupby("Year")[["Season"]].value_counts()
medals_per_game.sort_values(ascending=False, inplace=True)
medal_df = medals_per_game.head(3)  

age_count = italy_medals["Age"].value_counts()

italy_athletes = athlete_events.query("Team == 'Italy'")
all_italy_athletes = italy_athletes["Age"].value_counts()

italy_events = athlete_events.query("Team == 'Italy'")
sport_heights_mean = italy_events.groupby("Sport").agg({'Height': 'mean'})
sport_heights_mean = sport_heights_mean.dropna()
sport_heights_mean.sort_values(by="Height", inplace=True)

italy_events["BMI"] = italy_events["Weight"] / ((italy_events["Height"]/100)**2)

all_sports = medals['Sport'].unique()

athlete_events["BMI"] = athlete_events["Weight"] / ((athlete_events["Height"]/100)**2)

# Figures

# Italy medals
fig_italy_medals = px.bar(
    italy_medals_count,
    title="Italy medals",
    labels={"value": "Number of medals", 
            "variable": "Medals", 
            "count": "Number of medals"},
    template="simple_white"
    )

fig_italy_medals.update_layout(showlegend=False)

# Total number of medals awarded athletes per age group in Italy
fig_age_medals = px.histogram(
    age_count,
    x=age_count.index,
    y=age_count.values,
    nbins=len(age_count.index),
    labels={
        "Age": "Age Group", 
        "sum of y": "Number of Athletes"
    },
    title="The age groups of medal-awarded athletes in Italy",
    template="simple_white",
)

fig_age_medals.update_traces(hovertemplate=None)
fig_age_medals.update_layout(hovermode="x")
fig_age_medals.update_layout(yaxis_title="Number of medals")

# Total number of athletes per age group in Italy
fig_athletes_age_groups = px.histogram(
    all_italy_athletes,
    x=all_italy_athletes.index,
    y=all_italy_athletes.values,
    nbins=len(all_italy_athletes.index),
    title="Total number of athletes per age group in Italy",
    template="simple_white"
)

fig_athletes_age_groups.update_traces(hovertemplate=None)
fig_athletes_age_groups.update_layout(hovermode="x")
fig_athletes_age_groups.update_layout(yaxis_title="Number of athletes")

# Average Italian athletes' height per sport
fig_athlete_height_sport = px.bar(
    sport_heights_mean,
    x=sport_heights_mean.index,
    y="Height",
    title="Average Italian athlete height per sport",
    template="simple_white",
)

fig_athlete_height_sport.update_xaxes(tickangle=-45)
fig_athlete_height_sport.update_yaxes(range=[150, 193])

# BMI for Italian athletes per sport
fig_bmi_sport = px.box(
    italy_events,
    x="Sport",
    y="BMI",
    color="Sex",
    title="BMI index in different sports in Italy"
)

fig_bmi_sport.add_shape(
    type='line',
    x0="Rowing",
    x1="Rhythmic Gymnastics",
    y0=18.5,
    y1=18.5,
    line=dict(color='red', width=2, dash='dash'),
    name="Min range healthy BMI"
)

fig_bmi_sport.add_shape(
    type='line',
    x0="Rowing",
    x1="Rhythmic Gymnastics",
    y0=24.9,
    y1=24.9,
    line=dict(color='red', width=2, dash='dash'),
    name="Max range healthy BMI"
)

fig_bmi_sport.update_layout(
    xaxis = dict(
        tickangle=-45
    ),
)

# BMI for Italian athletes per age group

fig_bmi_age = px.box(
    italy_events,
    x="Age",
    y="BMI",
    color="Sex",
    hover_data=["Sport", "Year"],
    title="BMI index by age in Italy"
)

fig_bmi_age.add_shape(
    type='line',
    x0=italy_events['Age'].min(),
    x1=italy_events['Age'].max(),
    y0=18.5,
    y1=18.5,
    line=dict(color='red', width=2, dash='dash'),
    name="Min range healthy BMI"
)

fig_bmi_age.add_shape(
    type='line',
    x0=italy_events['Age'].min(),
    x1=italy_events['Age'].max(),
    y0=24.9,
    y1=24.9,
    line=dict(color='red', width=2, dash='dash'),
    name="Max range healthy BMI"
)


# BMI worldwide for Olympic Games

fig_bmi_worldwide = px.box(
    athlete_events,
    x="Sport",
    y="BMI",
    color="Sex",
    title="BMI index in different sports Worldwide"
)

fig_bmi_worldwide.add_shape(
    type='line',
    x0="Basketball",
    x1="Rhythmic Gymnastics",
    y0=18.5,
    y1=18.5,
    line=dict(color='red', width=2, dash='dash'),
    name="Min range healthy BMI"
)

fig_bmi_worldwide.add_shape(
    type='line',
    x0="Basketball",
    x1="Rhythmic Gymnastics",
    y0=24.9,
    y1=24.9,
    line=dict(color='red', width=2, dash='dash'),
    name="Max range healthy BMI"
)

fig_bmi_worldwide.update_layout(
    xaxis = dict(
        tickangle=-45
    ),
)
# App

app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Olympic Games Dashboard: Italy", className='text-center text-primary mx-3')
            ], width=12, style={'text-align': 'center'})
    ], className="mt-5"),
    
    # Gender distribution
    dbc.Row([
        dbc.Col([
            html.H4("Gender distribution historically, Italian athletes"),
            html.Img(src=dash.get_asset_url('gender_distribution_overall.png')) 
        ], xs=12, sm=11, md=10, lg=5, style={'text-align': 'center'}, className="mt-4"),        
        dbc.Col([
            html.H4("Gender distribution 2016, Italian athletes"),
            html.Img(src=dash.get_asset_url('gender_distribution_2016.png')) 
        ], xs=12, sm=11, md=10, lg=5, style={'text-align': 'center'}, className="mt-4"),
    ], justify='evenly'),
    
    
    # Medals per capita
    dbc.Row([
        dbc.Col([
            html.H5("Number of countries to display:"),
            dcc.Slider(id='top_countries_slider', min=3, max=206, step=1, value=10, marks=None),
            dcc.Graph(id='top_countries_graph')
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # Italy medals
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_italy_medals)
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # Italy medals by season
    dbc.Row([
        dbc.Col([
            html.Img(src=app.get_asset_url('italy_medals_by_year.png'), width='100%'),
        ], xs=12, sm=11, md=11, lg=10, style={'text-align': 'center'}),
    ], justify='center', className="mt-5"),
    
    # Data table of the best Italian years
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='best_italian_years',
                data=best_italian_years.to_dict('records'),
                columns=[{"name": i, "id": i} for i in best_italian_years.columns])
        ], xs=12, sm=11, md=11, lg=10, className="text-black-50")
    ], justify='center', className="mt-5"),
    
    # Age groups for Italy
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='medals_or_athletes_per_age_group',
                options=["Total number of athletes per age group in Italy", "The age groups of medal-awarded athletes in Italy"],
                value='Total number of athletes per age group in Italy'
            ),
            dcc.Graph(id='medals_or_athletes_per_age_group_graph')
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # Athlete heights per sport Italy
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_athlete_height_sport)
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # BMI for Italy
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='bmi_sport_or_age',
                options=["BMI index in different sports in Italy", "BMI index by age in Italy"],
                value='BMI index in different sports in Italy'
            ),
            dcc.Graph(id='bmi_sport_or_age_graph', style={'height': 800})
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # BMI Worldwide
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_bmi_worldwide, style={'height': 800})
        ], className="mt-5", xs=12, sm=11, md=11, lg=10)
    ], justify='center'),
    
    # Medals and age distribution for sports
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='sport_selector',
                options=[{'label': sport, 'value': sport} for sport in all_sports],
                value="Fencing"
            ),
            dcc.Graph(id='medal_graph')
        ], xs=12, sm=11, md=10, lg=5),
        dbc.Col([
            dcc.Dropdown(
                id='sport_selector2',
                options=[{'label': sport, 'value': sport} for sport in all_sports],
                value="Fencing"
            ),
            dcc.Graph(id='age_graph')
        ], xs=12, sm=11, md=10, lg=5)
    ], className="mt-5", justify='center'),
])

# Medals per capita worldwide
@callback(
    Output('top_countries_graph', 'figure'),
    Input('top_countries_slider', 'value')
)
def update_top_countries_per_capita(top_variable):
    top_countries = medals_per_capita.sort_values(ascending=False)[:top_variable].reset_index()
    top_countries['Population'] = merged_df.groupby('region')['Population  (2023)'].mean().loc[top_countries['region']].values
    top_countries.rename(columns={0: 'Medals'}, inplace=True)
    if top_variable == 206:
        fig_medals_per_capita = px.bar(top_countries, 
             x='region',
             y='Medals',
             color='Population',
             title=f'Number of Medals per Capita by Country: All countries',
             labels={'Medal': 'Medals per Capita', 'region': 'Country'},
             template='simple_white',
            )
    else:
        fig_medals_per_capita = px.bar(top_countries, 
             x='region',
             y='Medals',
             color='Population',
             title=f'Number of Medals per Capita by Country: Top {top_variable}',
             labels={'Medal': 'Medals per Capita', 'region': 'Country'},
             template='simple_white',
            )
    fig_medals_per_capita.update_layout(xaxis_title='Country', yaxis_title='Medals per Capita', showlegend=False)
    fig_medals_per_capita.update_traces(hovertemplate='Country: %{x}<br>Medals per Capita: %{y}<br>Population: %{marker.color:.2s}')
    return fig_medals_per_capita

# Athletes per age groups Italy
@callback(
    Output('medals_or_athletes_per_age_group_graph', 'figure'),
    Input('medals_or_athletes_per_age_group', 'value')
)
def medals_or_athletes_per_age_group(choice):
    if choice == "The age groups of medal-awarded athletes in Italy":
        return fig_age_medals
    else:
        return fig_athletes_age_groups

# Number of medals
@callback(
    Output('medal_graph', 'figure'),
    Input('sport_selector', 'value')
)
def update_medal_graph(selected_sport):
    medals_df = medals_count[medals_count['Sport'] == selected_sport]
    fig = px.bar(medals_df, 
                 x="Team", y="Count", 
                 color="Medal", 
                 title=f"Medals for {selected_sport}", 
                 template="simple_white", 
                 labels={"Team": "Country", "Count": "Number of medals"})
    fig.update_xaxes(tickangle=-45)
    return fig

# BMI index in Italy
@callback(
    Output('bmi_sport_or_age_graph', 'figure'),
    Input('bmi_sport_or_age', 'value')
)   
def update_bmi_graph(choice):
    if choice == "BMI index in different sports in Italy":
        return  fig_bmi_sport
    else:
        return fig_bmi_age

# Age distribution for sports in Italy
@callback(
    Output('age_graph', 'figure'),
    Input('sport_selector2', 'value')
)
def update_age_graph(selected_sport):
    sport_df = athlete_events[athlete_events['Sport'] == selected_sport]
    age_distribution = sport_df.groupby('Age').size().reset_index(name='Count')
    fig = px.bar(age_distribution, 
                 x="Age", y="Count", 
                 color="Age", 
                 title=f"Age distribution for {selected_sport}", 
                 template="simple_white",
                 labels={"Count": "Number of athletes"})
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8000)