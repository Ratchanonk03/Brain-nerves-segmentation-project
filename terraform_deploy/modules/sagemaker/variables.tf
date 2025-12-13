variable "model_image_path" {
    description = "Docker image for the SageMaker model"
    type        = string
    default     = "382416733822.dkr.ecr.us-east-1.amazonaws.com/linear-learner:latest"
}

variable "model_path" {
    description = "S3 path for the SageMaker model data"
    type        = string
    default     = "model/model.tar.gz"
}

variable "model_name" {
    description = "Name of the SageMaker model"
    type        = string
    default     = "linknet-34-model"
}

# S3 Bucket Names for SageMaker
variable "s3_preprocessing_bucket_name" {
    description = "Name of the S3 bucket for preprocessing data"
    type        = string
    default     = "preprocessing-bucket"
}

variable "s3_results_bucket_name" {
    description = "Name of the S3 bucket for results data"
    type        = string
    default     = "results-bucket"
}

# VPC Configuration for SageMaker
variable "vpc_private_subnet_id" {
    description = "ID of the private subnet in the VPC"
    type        = string
    default     = ""
}

variable "vpc_sagemaker_sg_id" {
    description = "ID of the security group for SageMaker"
    type        = string
    default     = ""
}