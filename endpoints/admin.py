from django.contrib import admin
from .models import Endpoint, MLAlgorithm, MLAlgorithmStatus, MLRequest


class EndPointAdmin(admin.ModelAdmin):
    fields = ['name', 'owner', 'created_at']
    list_display = ['name', 'owner', 'created_at']
    search_fields = ['name', 'owner']


class MLAlgorithmAdmin(admin.ModelAdmin):
    fields = ['name', 'description', 'code', 'version', 'owner', 'created_at', 'parent_endpoint']
    list_display = ['name', 'description', 'code', 'version', 'owner', 'created_at', 'parent_endpoint']
    search_fields = ['name', 'version', 'owner', 'parent_endpoint']


class MLAlgorithmStatusAdmin(admin.ModelAdmin):
    fields = ['status', 'active', 'created_by', 'created_at', 'parent_mlalgorithm']
    list_display = ['status', 'active', 'created_by', 'created_at', 'parent_mlalgorithm']
    search_fields = ['status', 'active', 'created_by', 'parent_mlalgorithm']


class MLRequestAdmin(admin.ModelAdmin):
    fields = ['input_data', 'full_response', 'response', 'feedback', 'created_at', 'parent_mlalgorithm']
    list_display = ['input_data', 'full_response', 'response', 'feedback', 'created_at', 'parent_mlalgorithm']
    search_fields = ['created_at', 'parent_mlalgorithm']


# Register your models here.
admin.register(Endpoint, EndPointAdmin)
admin.register(MLAlgorithm, MLAlgorithmAdmin)
admin.register(MLAlgorithmStatus)
admin.register(MLRequest, MLRequestAdmin)
