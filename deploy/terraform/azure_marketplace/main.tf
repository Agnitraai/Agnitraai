terraform {
  required_version = ">= 1.4.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.70"
    }
  }
}

provider "azurerm" {
  features {}
}

locals {
  env_base = [
    {
      name  = "AZURE_MARKETPLACE_RESOURCE_ID"
      value = var.marketplace_resource_id
    },
    {
      name  = "AZURE_MARKETPLACE_PLAN_ID"
      value = var.marketplace_plan_id
    },
    {
      name  = "AZURE_TENANT_ID"
      value = var.tenant_id
    },
    {
      name  = "AZURE_CLIENT_ID"
      value = var.client_id
    },
    {
      name  = "AGNITRA_ENV"
      value = "production"
    },
  ]

  env_extra = [
    for key, value in var.environment :
    {
      name  = key
      value = value
    } if value != null && value != ""
  ]
}

resource "azurerm_resource_group" "this" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "${var.name}-logs"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "this" {
  name                       = "${var.name}-env"
  location                   = azurerm_resource_group.this.location
  resource_group_name        = azurerm_resource_group.this.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
}

resource "azurerm_container_app" "runtime" {
  name                         = var.name
  resource_group_name          = azurerm_resource_group.this.name
  location                     = azurerm_resource_group.this.location
  container_app_environment_id = azurerm_container_app_environment.this.id

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "auto"
    traffic_weight {
      percentage = 100
      revision_suffix = "prod"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  secret {
    name  = "marketplace-client-secret"
    value = var.client_secret
  }

  template {
    container {
      name   = "api"
      image  = var.container_image
      cpu    = var.cpu
      memory = var.memory

      env = concat(
        local.env_base,
        [
          {
            name       = "AZURE_CLIENT_SECRET"
            secret_ref = "marketplace-client-secret"
          }
        ],
        local.env_extra,
      )
    }

    scale {
      min_replicas = var.min_replicas
      max_replicas = var.max_replicas

      rule {
        name = "http-scale"
        http {
          concurrent_requests = var.scale_http_concurrency
        }
      }
    }
  }
}
