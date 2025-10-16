# GCP Marketplace Deployment Module

This Terraform configuration deploys the Agnitra Marketplace runtime on Google
Cloud Run and prepares the credentials required for the Google Cloud Marketplace
SaaS metering API.

Resources provisioned:

- Required service APIs (Cloud Run, Cloud Resource Manager).
- A dedicated service account and key material stored in Secret Manager.
- Cloud Run service exposing the `/usage` endpoint and referencing the service
  account secret.
- IAM bindings that allow public invocation for testing or proof-of-concept
  setups.

## Usage

```hcl
module "agnitra_marketplace" {
  source = "./deploy/terraform/gcp_marketplace"

  project_id                = "my-gcp-project"
  region                    = "us-central1"
  container_image           = "us-central1-docker.pkg.dev/my-gcp-project/agnitra/marketplace:latest"
  marketplace_service_name  = "services/123456789012"
  marketplace_sku_id        = "AAAA-BBBB-1234"
}
```

Apply the module and register the resulting `/usage` URL with the Google Cloud
Marketplace SaaS integration workflow so that usage reports flow directly to
Google's billing infrastructure.
