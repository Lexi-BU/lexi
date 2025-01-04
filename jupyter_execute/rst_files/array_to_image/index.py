#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lexi_xray.lexi import array_to_image
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D array
input_array = np.random.rand(100, 100)

# Print the shape of the input array
# The shape should be (100, 100)
print(input_array.shape)

