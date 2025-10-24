{{- if .Values.serviceAccount.create }}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "agnitra-marketplace.fullname" . }}
  labels:
    {{- include "agnitra-marketplace.labels" . | nindent 4 }}
{{- end }}
