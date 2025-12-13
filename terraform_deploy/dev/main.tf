terraform {
    backend "s3" {
        bucket         = "devops-directive-terraform-state-bucket"
        key            = "tf-infra/terraform.tfstate"
        region         = "us-east-1"
        dynamodb_table = "terraform-state-locking"
        encrypt        = true
    }

    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 6.0"
        }
    }
}

provider "aws" {
  region = "us-east-1"
}

module "s3" {
    source = "../modules/s3"

    # Input variables
    results_bucket_name       = "results-bucket"
    preprocessing_bucket_name = "preprocessing-bucket"
    versioning_status         = "Disabled"
    sse_algorithm             = "AES256"
}

data "aws_s3_object" "model_path" {
    bucket = "model-bucket01-v1"
    key    = "linknet-bare/model-v5.tar.gz"
}

data "aws_ecr_image" "model_image" {
    repository_name = "medical-ai/model"
    image_tag       = "temp-production"
}

module "sagemaker" {
    source = "../modules/sagemaker"

    # Input variables
    model_image_path     = data.aws_ecr_image.model_image.image_uri
    model_path           = "s3://${data.aws_s3_object.model_path.id}"
    model_name           = "linknet-34-dev"

    s3_preprocessing_bucket_name = module.s3.preprocessing_bucket_name
    s3_results_bucket_name       = module.s3.results_bucket_name
}