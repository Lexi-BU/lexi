#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lexi.lexi import get_exposure_maps

exposure_maps_dict = get_exposure_maps(
    time_range=["2025-03-02 08:04:00", "2025-03-08 23:43:00"],
    ra_range=[190, 310],
    dec_range=[-33, 3],
    ra_res=0.5,
    dec_res=0.5,
    save_exposure_map_file=False,
    save_exposure_map_image=True,
    verbose=False,
    array_to_image_kwargs={"display": True}
)

print(exposure_maps_dict.keys())

