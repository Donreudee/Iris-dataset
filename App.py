Jelinnnnnnn
bluewhale_01
ไม่ระบุ

Jelinnnnnnn จอมซ่าปรากฏตัว — 13:25
สมริน มาแล้ว — 13:25
𝓝𝓾𝓽 จอมซ่าปรากฏตัว — 13:26
𝓝𝓾𝓽 — 13:27
# Project2_GroupXX
# Student IDs and Names:
# 6XXXXXXXX Student Name 1
# 6XXXXXXXX Student Name 2
# 6XXXXXXXX Student Name 3
ขยาย
message.txt
6 KB
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ขยาย
message.txt
17 KB
streamlit==1.30.0
pandas==2.1.3
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
plotly==5.18.0
6531501040 Nattawut Pongkasem
𝓝𝓾𝓽 — 13:34
# app.py - Streamlit App for Iris Species Classifier
# Project2_GroupXX

import streamlit as st
import pandas as pd
import numpy as np
ขยาย
message.txt
12 KB
𝓝𝓾𝓽 — 14:28
# app.py - Streamlit App for Iris Species Classifier
# Project2_GroupXX

import streamlit as st
import pandas as pd
import numpy as np
ขยาย
message.txt
17 KB
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
ขยาย
message.txt
3 KB
ชานมไข่มุก — 14:47
ประเภทไฟล์ที่แนบ: unknown
best_model.pkl
1.02 KB
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
ขยาย
requirements.txt
1 KB
﻿
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Iris Dataset Visualization",
    page_icon="🌸",
    layout="wide"
)

# Function to load the dataset
@st.cache_data
def load_data():
    try:
        # Try to load from file
        data = pd.read_csv('iris.csv')
        return data
    except:
        # If not available, return a sample or error
        st.warning("Could not load the dataset. Using a sample dataset instead.")
        # Create a minimal sample
        return pd.DataFrame({
            'sepal_length': [5.1, 6.2, 7.3],
            'sepal_width': [3.5, 2.9, 2.9],
            'petal_length': [1.4, 4.3, 6.3],
            'petal_width': [0.2, 1.3, 1.8],
            'species': ['setosa', 'versicolor', 'virginica']
        })

# Load data
df = load_data()

# Main app title
st.title("🌸 Iris Dataset Visualization")
st.markdown("### Pairplot of Iris Features by Species")

# Create and display pairplot
fig = plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species', markers=["o", "o", "o"], 
             plot_kws={'alpha': 0.7, 's': 30}, 
             diag_kind='kde', 
             diag_kws={'alpha': 0.5, 'shade': True, 'bw_adjust': 0.8})
plt.tight_layout()
st.pyplot(fig)

# Add a brief explanation
st.markdown("""
### About This Visualization

This pairplot shows the relationships between the four features of the Iris dataset:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each species is represented by a different color:
- Blue: Iris Setosa
- Orange: Iris Versicolor
- Green: Iris Virginica

The diagonal plots show the distribution of each feature for each species, while the scatter plots show the relationships between pairs of features.

This visualization clearly shows how the three species cluster differently based on their measurements, particularly for petal dimensions.
""")

# Footer
st.markdown("---")
st.markdown("Iris Flower Classification for Horticultural Business Decision Support")
message.txt
3 KB