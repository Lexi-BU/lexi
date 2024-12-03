{% if fullname %}
.. _{{ fullname }}:

{{ fullname.split('.')[-1] }}
=============================

**Docstring:**

{{ docstring }}
{% endif %}
