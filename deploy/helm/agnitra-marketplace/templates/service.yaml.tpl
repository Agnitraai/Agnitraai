apiVersion: v1
kind: Service
metadata:
  name: {{ include "agnitra-marketplace.fullname" . }}
  labels:
    {{- include "agnitra-marketplace.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  selector:
    {{- include "agnitra-marketplace.selectorLabels" . | nindent 4 }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.targetPort }}
      protocol: TCP
      name: http
