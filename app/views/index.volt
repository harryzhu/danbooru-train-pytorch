<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Console">
        <meta name="author" content="Harry">
        <title></title>

        {{ stylesheet_link("assets/bootstrap/css/bootstrap.min.css") }}
        {{ stylesheet_link("assets/css/status.css") }}
        {{ javascript_include("assets/bootstrap/js/jquery-2.1.1.min.js") }}
        {{ javascript_include("assets/bootstrap/js/bootstrap.min.js") }}
    </head>

    <body>
        {{ partial("partials/nav") }}

        <div class="container-fluid content">
            {{ content() }}
            <div class="clearfix"></div>
        </div>
        <div class="container-fluid footer">
            {{ partial("partials/footer") }}
        </div>
    </body>
</html>