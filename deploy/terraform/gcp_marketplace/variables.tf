variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name for the Cloud Run service."
  type        = string
  default     = "agnitra-marketplace"
}

variable "container_image" {
  description = "Container image reference stored in Artifact Registry or GCR."
  type        = string
}

variable "marketplace_service_name" {
  description = "Marketplace service resource name (e.g., services/1234567890)."
  type        = string
}

variable "marketplace_sku_id" {
  description = "Marketplace SKU identifier used for metering."
  type        = string
}

variable "environment" {
  description = "Additional environment variables for the container."
  type        = map(string)
  default     = {}
}

variable "min_scale" {
  description = "Minimum number of Cloud Run instances."
  type        = number
  default     = 1
}

variable "max_scale" {
  description = "Maximum number of Cloud Run instances."
  type        = number
  default     = 5
}
