{{/*
Account-specific names. Set aws.accountId once (helm --set aws.accountId=123456789012); every ARN, ECR
repository, bucket and queue below is derived from it unless overridden explicitly in values.yaml.
*/}}
{{- define "argus.accountId" -}}
{{- required "set aws.accountId (helm --set aws.accountId=<12-digit AWS account id>)" .Values.aws.accountId -}}
{{- end -}}

{{- define "argus.image" -}}
{{- $repo := .Values.image.repository | default (printf "%s.dkr.ecr.%s.amazonaws.com/argus/operator" (include "argus.accountId" .) .Values.aws.region) -}}
{{- printf "%s:%s" $repo .Values.image.tag -}}
{{- end -}}

{{- define "argus.checkpointBucket" -}}
{{- .Values.aws.checkpointBucket | default (printf "argus-checkpoints-%s" (include "argus.accountId" .)) -}}
{{- end -}}

{{- define "argus.riskEventsQueueUrl" -}}
{{- .Values.aws.riskEventsQueueUrl | default (printf "https://sqs.%s.amazonaws.com/%s/argus-risk-events" .Values.aws.region (include "argus.accountId" .)) -}}
{{- end -}}

{{- define "argus.irsaRoleArn" -}}
{{- .Values.serviceAccount.irsaRoleArn | default (printf "arn:aws:iam::%s:role/argus-operator-irsa" (include "argus.accountId" .)) -}}
{{- end -}}
