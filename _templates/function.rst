{% if fullname %}
.. _{{ fullname }}:

{{ fullname.split('.') }}
=============================

**Docstring:**

{{ docstring }}
{% endif %}
