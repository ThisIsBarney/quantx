{{/*
Expand the name of the chart.
*/}}
{{- define "quantx.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "quantx.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label.
*/}}
{{- define "quantx.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "quantx.labels" -}}
helm.sh/chart: {{ include "quantx.chart" . }}
{{ include "quantx.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "quantx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "quantx.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Job store URL — auto-derive Redis URL when redis sub-chart is enabled.
*/}}
{{- define "quantx.jobStoreUrl" -}}
{{- if .Values.redis.enabled -}}
redis://{{ include "quantx.fullname" . }}-redis-master:6379/0
{{- else if .Values.jobStore.url -}}
{{ .Values.jobStore.url }}
{{- else -}}
{{- end }}
{{- end }}
