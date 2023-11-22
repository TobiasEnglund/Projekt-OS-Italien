# Add this to the code to get italian flag as background,
# Don't forget to add children= before all the elements

# Define the path to the PNG image
image_path = './assets/Flag_of_Italy.png'  # Replace with the actual path

app.layout = html.Div(
    style={
        'background-image': f'url("{image_path}")',
        'background-size': '100% auto',  # Cover width, auto height
        'background-repeat': 'repeat-y',  # Repeat vertically
        'height': '100vh',  # Set height to 100% of the viewport height
    },
    children=[
        html.H1("Your Dash App Content Goes Here"),
        # Add your Dash components/content here
    ]
)