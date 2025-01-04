#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lexi_xray.lexi import get_lexi_data

df_lexi = get_lexi_data(
    time_range=["2025-03-02 08:50:00", "2025-03-02 09:23:00"],
    verbose=False
)

print(df_lexi.head())

