# AWS Marketplace Deployment Module

This Terraform module provisions an ECS Fargate service that runs the Agnitra
Marketplace API container and exposes the `/usage` billing endpoint through an
Application Load Balancer. The stack includes:

- A dedicated VPC with public subnets and internet gateway.
- Application Load Balancer and target group for the runtime service.
- ECS cluster, task definition, and service with autoscaling policies.
- CloudWatch log group wired to the container.

## Usage

```hcl
module "agnitra_marketplace" {
  source = "./deploy/terraform/aws_marketplace"

  region                     = "us-east-1"
  container_image            = "123456789012.dkr.ecr.us-east-1.amazonaws.com/agnitra-marketplace:latest"
  marketplace_product_code   = "abcd1234example"
  marketplace_usage_dimension = "RUNTIME_OPTIMIZATION_HOURS"
  environment = {
    GCP_MARKETPLACE_SERVICE_NAME = "services/..."
  }
}
```

After `terraform apply`, the module outputs the load balancer DNS name and a
ready-to-use `/usage` endpoint URL. Wire the endpoint into your AWS Marketplace
SaaS contract using the AWS SaaS metering callbacks.
