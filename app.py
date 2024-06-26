import streamlit as st
from PIL import Image, ImageOps
import io
from generator import generate_img

st.set_page_config(
    page_title="Image Colorization App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to convert image to grayscale

# Define the fixed top navigation bar
st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] {
        visibility: hidden;
    }
    .fixed-nav-bar {
        position: relative;
        top: 0;
        padding:0px;
        background-color: transparent;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    body{
        padding-right:0px;
    }
    .content {
    }
    .gen-image {
        display:block;
        
    }
    body {
        padding:0;
        margin:0;
    }
    </style>
    <div class="fixed-nav-bar">
        <h2>Image Processing App</h2>
    </div>
    """, unsafe_allow_html=True)

# Content starts here
col20,col21 = st.columns([1.2,1])
with col20:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.header('About this Project')
    st.write("""
        The colorization of black and white images is a process that breathes new life into grayscale photos by adding color, enhancing their visual appeal and historical value. 
        Traditionally, this task was performed manually by artists who meticulously added colors based on their judgment and available information about the original scene. This method, while effective, is labor-intensive and demands a high level of skill and significant time investment.
        The emergence of digital technologies, particularly in the field of machine learning and artificial intelligence, has brought a paradigm shift in image colorization. Automated techniques have evolved to address the limitations of manual methods, offering faster and more precise colorization. Among these advanced techniques, generative adversarial networks (GANs) have proven to be particularly effective. GANs have the capability to learn complex patterns and produce highly realistic colorized images, making them a powerful tool for this application.
    """)


    st.subheader('Gan Architecture')


    col5 , col6 = st.columns([1,1])
    with col5:
        st.image ('temp_image/gen_visualkeras.jpg')
    with col6:
        st.write('''The generator model for image colorization is structured with various layers, each playing a crucial role in shaping the network's architecture. 
             ''')
    st.write('''It comprises 9 convolutional layers distributed across three blocks, where each block consists of 3 convolutional layers. Additionally, a bottleneck layer condenses the extracted features, while two concatenation layers merge bottleneck outputs with earlier feature maps. Furthermore, the model employs 6 transposed convolutional layers, divided into two sets of 3 layers each, for upscaling purposes.''')
    st.write('''The workflow of the model initiates with an initial feature extraction phase driven by the convolutional layers. These layers progressively learn hierarchical representations of features from grayscale images, capturing essential patterns and details. Subsequently, the bottleneck layer compresses these features into a more manageable form, facilitating efficient processing and information retrieval. Concatenation layers then integrate bottleneck outputs with feature maps from previous layers, preserving vital spatial information essential for accurate colorization.''')






with col21:
    st.header('Upload Image')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown('<div class="flex_prop">', unsafe_allow_html=True)
    generate_button = st.button('Generate Image')


    # Section about the website
    

    # Image upload section
        
    if uploaded_file is not None:
            # Display the uploaded image
            col3,col4=st.columns(2)
            with col3:
                st.subheader('Uploaded Image')
                st.markdown('<div class=gen-image>', unsafe_allow_html=True)
                org_image = Image.open(uploaded_file)
                image = org_image.resize((120,120))
                st.image(org_image, caption='Uploaded Image.',output_format='auto')
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate button
            with col4:
            
                if generate_button:
                    st.subheader('Colorizied Image')
                    st.markdown('<div class=gen-image>', unsafe_allow_html=True)
                    generated_image = generate_img(org_image).resize((150,150))
                    st.image(generated_image, caption='Generated Image.')
                    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
