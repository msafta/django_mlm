# /endpoints/views.py file

"""
For each model, we created a view which will allow to retrieve single object or list of objects.
We will not allow to create or modify Endpoints, MLAlgorithms by REST API.
The code to to handle creation of new ML related objects will be on server side.

We will allow to create MLAlgorithmStatus objects by REST API.
We don't allow to edit statuses for ML algorithms as we want to keep all status history.

We allow to edit MLRequest objects, however only feedback field (please take a look at serializer definition).
"""

from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from rest_framework import mixins
from django.db import transaction
from rest_framework.exceptions import APIException

from endpoints.models import Endpoint
from endpoints.serializers import EndpointSerializer

from endpoints.models import MLAlgorithm
from endpoints.serializers import MLAlgorithmSerializer

from endpoints.models import MLAlgorithmStatus
from endpoints.serializers import MLAlgorithmStatusSerializer

from endpoints.models import MLRequest
from endpoints.serializers import MLRequestSerializer

import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
from ML.registry import MLRegistry
from django_mlm.wsgi import registry


class EndpointViewSet(mixins.RetrieveModelMixin,
                      mixins.ListModelMixin,
                      viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(mixins.RetrieveModelMixin,
                         mixins.ListModelMixin,
                         viewsets.GenericViewSet):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
                                                    created_at__lt=instance.created_at,
                                                    active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


class MLAlgorithmStatusViewSet(mixins.RetrieveModelMixin,
                               mixins.ListModelMixin,
                               viewsets.GenericViewSet,
                               mixins.CreateModelMixin):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)
        except Exception as e:
            raise APIException(str(e))


class MLRequestViewSet(mixins.RetrieveModelMixin,
                       mixins.ListModelMixin,
                       viewsets.GenericViewSet,
                       mixins.UpdateModelMixin):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()


class PredictView(views.APIView):
    """
    View for predictions that can accept POST requests with JSON data and forward it to the
    correct ML algorithm.
    """
    def post(self, request, endpoint_name, format=None):

        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status, status__active=True)

        if algorithm_version is not None:
            algs = algs.filter(version = algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(algs) != 1 and algorithm_status != "ab_testing":
            print(len(algs))
            return Response(
                {"status": "Error", "message": "ML algorithm selection is ambiguous. Please specify algorithm version."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if rand() < 0.5 else 1

        # print(alg_index)
        # print(algs[0])
        # print(algs[0].id)
        # print(registry.endpoints)

        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id

        return Response(prediction)
