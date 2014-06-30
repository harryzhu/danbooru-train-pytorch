<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Console">
        <meta name="author" content="Harry">
        <title></title>

        <link href="assets/bootstrap/css/bootstrap.min.css" rel="stylesheet">

    </head>

	<body>
<?php echo $this->partial('partials/nav'); ?>

		<?php echo $this->getContent(); ?>
	
        <script src="assets/bootstrap/js/jquery-2.1.1.min.js"></script>
        <script src="assets/bootstrap/js/bootstrap.js"></script>

</body>
</html>