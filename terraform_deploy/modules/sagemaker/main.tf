data "aws_iam_role" "lab_role" {
    name = "LabRole"
}

resource "aws_sagemaker_model" "model" {
    name                 = var.model_name
    execution_role_arn   = data.aws_iam_role.lab_role.arn
    primary_container {
        image          = var.model_image_path
        model_data_url = var.model_path
        environment = {
            PREPROCESSED_BUCKET=var.s3_preprocessing_bucket_name
            RESULTS_BUCKET=var.s3_results_bucket_name
            }
    }

    dynamic "vpc_config" {
        for_each = var.vpc_private_subnet_id != "" && var.vpc_sagemaker_sg_id != "" ? [1] : []
        content {
            subnets            = [var.vpc_private_subnet_id]
            security_group_ids = [var.vpc_sagemaker_sg_id]
        }
    }

}