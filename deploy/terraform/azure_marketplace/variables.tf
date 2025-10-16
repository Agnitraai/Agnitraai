variable "resource_group_name" {
  description = "Resource group used for the container app."
  type        = string
  default     = "rg-agnitra-marketplace"
}

variable "location" {
  description = "Azure region for the deployment."
  type        = string
  default     = "eastus"
}

variable "name" {
  description = "Container App name."
  type        = string
  default     = "agnitra-marketplace"
}

variable "container_image" {
  description = "Container image reference for the Agnitra runtime."
  type        = string
}

variable "marketplace_resource_id" {
  description = "Azure marketplace resource ID used for metering."
  type        = string
}

variable "marketplace_plan_id" {
  description = "Marketplace plan ID associated with the SaaS offer."
  type        = string
}

variable "client_id" {
  description = "Azure AD application (client) ID with marketplace metering access."
  type        = string
}

variable "client_secret" {
  description = "Client secret used to fetch Azure marketplace tokens."
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Azure AD tenant ID."
  type        = string
}

variable "environment" {
  description = "Additional environment variables injected into the container."
  type        = map(string)
  default     = {}
}

variable "min_replicas" {
  description = "Minimum number of container app replicas."
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of container app replicas."
  type        = number
  default     = 5
}

variable "scale_http_concurrency" {
  description = "Target concurrent HTTP requests per replica."
  type        = number
  default     = 50
}

variable "cpu" {
  description = "CPU cores allocated to each replica."
  type        = number
  default     = 0.5
}

variable "memory" {
  description = "Memory allocated to each replica."
  type        = string
  default     = "1Gi"
}
