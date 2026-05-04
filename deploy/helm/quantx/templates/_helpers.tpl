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
Job store URL resolution — priority order:
  1. redis.enabled=true  → Redis sub-chart service (Bitnami naming: <release>-redis-master)
  2. jobStore.url set    → use verbatim
  3. jobStore.backend    → derive canonical URL from backend type:
       "redis"  → redis://<release>-redis-master:6379/0
       "sqlite" → sqlite:///data/jobs.db
       "memory" → "" (no env var injected; in-process default)

Using .Release.Name (not fullname) matches the Bitnami redis sub-chart's own
service naming convention, which is always <release>-redis-master regardless of
any nameOverride on the parent chart.
*/}}
{{- define "quantx.jobStoreUrl" -}}
{{- if .Values.redis.enabled -}}
redis://{{ .Release.Name }}-redis-master:6379/0
{{- else if .Values.jobStore.url -}}
{{ .Values.jobStore.url | trim }}
{{- else if eq .Values.jobStore.backend "redis" -}}
redis://{{ .Release.Name }}-redis-master:6379/0
{{- else if eq .Values.jobStore.backend "sqlite" -}}
sqlite:///data/jobs.db
{{- else -}}
{{- end }}
{{- end }}
