"""Django_FeatureTools URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path
# from ft_model.views import get_results, no_page, selected_features, select_tables, variables_type, model_parameters

from ft_model.views import no_page, select_tables, model_parameters, variables_type, get_results,select_features, plot_wordcloud

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', select_tables),
    path('select_tables/', select_tables),
    path('variables_type/', variables_type),
    path('get_results/', get_results),
    # path('selected_features/', selected_features),
    path('model_parameters/', model_parameters),
    path('select_features/', select_features),
    path('plot_wordcloud/', plot_wordcloud),
    # url(r'^\w+', no_page),
    re_path(r'^\w+', no_page),
]
