apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "agnitra-marketplace.fullname" . }}
  labels:
    {{- include "agnitra-marketplace.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "agnitra-marketplace.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "agnitra-marketplace.selectorLabels" . | nindent 8 }}
      {{- with .Values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      {{- if .Values.serviceAccount.create }}
      serviceAccountName: {{ include "agnitra-marketplace.fullname" . }}
      {{- else if .Values.serviceAccount.name }}
      serviceAccountName: {{ .Values.serviceAccount.name }}
      {{- end }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: api
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          env:
            {{- range $key, $value := .Values.env }}
            {{- if ne $key "EXTRA_ENV" }}
            - name: {{ $key }}
              value: "{{ $value }}"
            {{- end }}
            {{- end }}
            {{- range .Values.env.EXTRA_ENV }}
            - name: {{ .name }}
              value: {{ .value | quote }}
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          livenessProbe:
            {{- toYaml .Values.livenessProbe | nindent 12 }}
          readinessProbe:
            {{- toYaml .Values.readinessProbe | nindent 12 }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          {{- if .Values.extraVolumeMounts }}
          volumeMounts:
            {{- range .Values.extraVolumeMounts }}
            - name: {{ .name }}
              mountPath: {{ .mountPath }}
              {{- if .readOnly }}
              readOnly: {{ .readOnly }}
              {{- end }}
            {{- end }}
          {{- end }}
      {{- if .Values.extraVolumes }}
      volumes:
        {{- range .Values.extraVolumes }}
        - name: {{ .name }}
          {{- toYaml .config | nindent 10 }}
        {{- end }}
      {{- end }}
