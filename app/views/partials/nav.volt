<nav class="navbar navbar-default navbar-fixed-top" role="navigation">
    <!-- Brand and toggle get grouped for better mobile display -->
    <div class="navbar-header">
        <?php echo Phalcon\Tag::linkTo(array('', 'Status','class'=>'navbar-brand')); ?>
    </div>

    <!-- Collect the nav links, forms, and other content for toggling -->
    <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
        <ul class="nav navbar-nav">
            <li class="dropdown">
                <a href="#" class="dropdown-toggle" data-toggle="dropdown">Services <b class="caret"></b></a>
                <ul class="dropdown-menu">
                    <li class="dropdown-header">Compute and Networking</li>
                    <li><a href="#"><?php echo Phalcon\Tag::linkTo('server', 'Server'); ?></a></li>
                    <li><a href="#">Cluster</a></li>
                    <li class="divider"></li>
                    <li class="dropdown-header">Storage and CDN</li>
                    <li><a href="#">DFS</a></li>
                    <li><a href="#">Shared Folder</a></li>
                    <li class="divider"></li>
                    <li class="dropdown-header">Database</li>
                    <li><a href="#">SQL Server</a></li>                    
                    <li class="divider"></li> 
                    <li class="dropdown-header">Log</li>
                    <li><a href="#">Event Log</a></li>
                    <li><a href="#">IIS Log</a></li>
                </ul>
            </li>
        </ul>





        <ul class="nav navbar-nav navbar-right">
            <li><a href="#">User</a></li>
            <li class="dropdown">
                <a href="#" class="dropdown-toggle" data-toggle="dropdown">Location <b class="caret"></b></a>
                <ul class="dropdown-menu">
                    <li><a href="#">BJ1</a></li>
                    <li><a href="#">BJ</a></li>
                    <li class="divider"></li>
                    <li><a href="#">SHA</a></li>
                    <li><a href="#">SH</a></li>
                </ul>
            </li>
        </ul>
    </div><!-- /.navbar-collapse -->
</nav>

