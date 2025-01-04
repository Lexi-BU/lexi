#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lexi_xray.lexi import make_lexi_images

lexi_images_dict = make_lexi_images(
    time_range=["2025-03-04 08:53:41", "2025-03-04 09:23:41"],
    ra_range=[220, 240],
    dec_range=[-30, -15],
    ra_res=1,
    dec_res=1,
    background_correction_on=False,
    save_exposure_map_file=False,
    save_sky_backgrounds_file=False,
    save_exposure_map_image=False,
    save_sky_backgrounds_image=False,
    save_lexi_images=True,
    verbose=False,
    array_to_image_kwargs={"display": True}
)

print(lexi_images_dict.keys())

