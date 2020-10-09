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

from django.db import transaction
from endpoints.models import ABTest
from endpoints.serializers import ABTestSerializer

from django.db.models import F
import datetime


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
            algs = algs.filter(version=algorithm_version)

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


class ABTestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
                    mixins.CreateModelMixin, mixins.UpdateModelMixin):
    """
    The ABTestViewSet view allows the user to create new objects.
    The perform_create method creates the ABTest object and two new statuses for ML algorithms.
    The new statuses are set to ab_testing.
    We will add also a view to stop the A/B test.
    """

    serializer_class = ABTestSerializer
    queryset = ABTest.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save()
                # update status for first algorithm

                status_1 = MLAlgorithmStatus(status = "ab_testing", created_by=instance.created_by,
                                             parent_mlalgorithm=instance.parent_mlalgorithm_1, active=True)
                status_1.save()
                deactivate_other_statuses(status_1)
                # update status for second algorithm
                status_2 = MLAlgorithmStatus(status="ab_testing", created_by=instance.created_by,
                                             parent_mlalgorithm=instance.parent_mlalgorithm_2, active=True)
                status_2.save()
                deactivate_other_statuses(status_2)

        except Exception as e:
            raise APIException(str(e))


class StopABTestView(views.APIView):
    """
    The StopABTestView stops the A/B test and compute the accuracy (ratio of correct responses) for each algorithm.
    The algorithm with higher accurcy is set as production algorithm, the other algorithm is saved with testing status.
    """
    def post(self, request, ab_test_id, format=None):

        try:
            ab_test = ABTest.objects.get(pk=ab_test_id)

            if ab_test.ended_at is not None:
                return Response({"message": "AB Test already finished."})

            date_now = datetime.datetime.now()
            # alg #1 accuracy
            all_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()

            correct_responses_1 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_1,
                                                           created_at__gt=ab_test.created_at,
                                                           created_at__lt=date_now,
                                                           response=F('feedback')).count()
            accuracy_1 = correct_responses_1 / float(all_responses_1)
            print(all_responses_1, correct_responses_1, accuracy_1)

            # alg #2 accuracy
            all_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                       created_at__gt=ab_test.created_at,
                                                       created_at__lt=date_now).count()
            correct_responses_2 = MLRequest.objects.filter(parent_mlalgorithm=ab_test.parent_mlalgorithm_2,
                                                           created_at__gt=ab_test.created_at,
                                                           created_at__lt=date_now,
                                                           response=F('feedback')).count()
            accuracy_2 = correct_responses_2 / float(all_responses_2)
            print(all_responses_2, correct_responses_2, accuracy_2)

            # select algorithm with higher accuracy
            alg_id_1, alg_id_2 = ab_test.parent_mlalgorithm_1, ab_test.parent_mlalgorithm_2
            # swap
            if accuracy_1 < accuracy_2:
                alg_id_1, alg_id_2 = alg_id_2, alg_id_1

            status_1 = MLAlgorithmStatus(status="production", created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_1, active=True)
            status_1.save()
            deactivate_other_statuses(status_1)
            # update status for second algorithm
            status_2 = MLAlgorithmStatus(status="testing", created_by=ab_test.created_by,
                                         parent_mlalgorithm=alg_id_2, active=True)
            status_2.save()
            deactivate_other_statuses(status_2)

            summary = "Algorithm #1 accuracy: {}, Algorithm #2 accuracy: {}".format(accuracy_1, accuracy_2)
            ab_test.ended_at = date_now
            ab_test.summary = summary
            ab_test.save()

        except Exception as e:
            return Response({"status": "Error", "message": str(e)},
                            status=status.HTTP_400_BAD_REQUEST
            )
        return Response({"message": "AB Test finished.", "summary": summary})
