"""
WSGI config for django_mlm project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_mlm.settings')

application = get_wsgi_application()

"""
Add ML algorithms to the registry when the server is starting.
"""

# import os
# from django.core.wsgi import get_wsgi_application
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
# application = get_wsgi_application()

# ML registry
import inspect
from ML.registry import MLRegistry
from ML.income_classifier.random_forest import RandomForestClassifier

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    rf = RandomForestClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                           algorithm_object=rf,
                           algorithm_name="random forest",
                           algorithm_status="production",
                           algorithm_version="0.0.2",
                           owner="Marius",
                           algorithm_description="Random Forest with simple pre- and post-processing",
                           algorithm_code=inspect.getsource(RandomForestClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
