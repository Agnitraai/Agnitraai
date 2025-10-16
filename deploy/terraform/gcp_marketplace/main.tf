terraform {
  required_version = ">= 1.4.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.70"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "current" {}

locals {
  project_number = data.google_project.current.number
  env_base = [
    {
      name  = "GCP_PROJECT_NUMBER"
      value = data.google_project.current.number
    },
    {
      name  = "GCP_MARKETPLACE_SERVICE_NAME"
      value = var.marketplace_service_name
    },
    {
      name  = "GCP_MARKETPLACE_SKU_ID"
      value = var.marketplace_sku_id
    },
    {
      name  = "GCP_MARKETPLACE_SERVICE_ACCOUNT"
      value = "/var/secrets/google/service-account.json"
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

resource "google_project_service" "run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "cloudresourcemanager" {
  service = "cloudresourcemanager.googleapis.com"
}

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-sa"
  display_name = "Agnitra Marketplace Runtime"
}

resource "google_service_account_key" "runtime" {
  service_account_id = google_service_account.runtime.name
  keepers = {
    latest = timestamp()
  }
}

resource "google_secret_manager_secret" "sa" {
  secret_id = "${var.service_name}-marketplace-sa"

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "sa" {
  secret      = google_secret_manager_secret.sa.id
  secret_data = google_service_account_key.runtime.private_key
}

resource "google_cloud_run_service" "runtime" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.runtime.email
      containers {
        image = var.container_image
        ports {
          name           = "http1"
          container_port = 8080
        }
        env = concat(local.env_base, local.env_extra)
        volume_mounts {
          name       = "google-creds"
          mount_path = "/var/secrets/google"
          read_only  = true
        }
      }
      volumes {
        name = "google-creds"
        secret {
          secret_name = google_secret_manager_secret.sa.secret_id
          items {
            key  = "latest"
            path = "service-account.json"
          }
        }
      }
    }
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = tostring(var.min_scale)
        "autoscaling.knative.dev/maxScale" = tostring(var.max_scale)
      }
    }
  }

  traffic {
    percent = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "invoker" {
  service  = google_cloud_run_service.runtime.name
  location = google_cloud_run_service.runtime.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
