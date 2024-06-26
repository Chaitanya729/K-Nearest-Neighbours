#include "TreeIndex.h"

int max_cols = 784;
/**
 * @fn DataVector::DataVector(int dimension)
 * @brief Constructor for the DataVector class.
 * @param dimension The dimension of the vector.
 */
DataVector::DataVector(int dimension){
    v.resize(dimension);
}

/**
 * @fn DataVector::~DataVector()
 * @brief Destructor for the DataVector class.
 */
DataVector::~DataVector()
{
    vector<double>().swap(v);
}

/**
 * @fn DataVector::DataVector(const DataVector& other)
 * @brief Copy constructor for the DataVector class.
 * @param other The DataVector to copy.
 */
DataVector::DataVector(const DataVector& other)
{
    v = other.v;
}

/**
 * @fn DataVector & DataVector::operator=(const DataVector &other)
 * @brief Assignment operator for the DataVector class.
 * @param other The DataVector to assign from.
 * @return A reference to this DataVector.
 */
DataVector & DataVector::operator=(const DataVector &other)
{
    v = other.v;
    return *this;
}

/**
 * @fn void DataVector::setDimension(int dimension)
 * @brief Sets the dimension of the DataVector.
 * @param dimension The new dimension.
 */
void DataVector::setDimension(int dimension)
{
    v.resize(dimension);
}

/**
 * @fn DataVector DataVector::operator+(const DataVector &other)
 * @brief Addition operator for the DataVector class.
 * @param other The DataVector to add.
 * @return The resulting DataVector.
 */
DataVector DataVector::operator+(const DataVector &other)
{
    DataVector result(v.size());
    for (int i = 0; i < v.size(); i++)
    {
        result.v[i] = v[i] + other.v[i];
    }
    return result;
}

/**
 * @fn DataVector DataVector::operator-(const DataVector &other)
 * @brief Subtraction operator for the DataVector class.
 * @param other The DataVector to subtract.
 * @return The resulting DataVector.
 */
DataVector DataVector::operator-(const DataVector &other)
{
    DataVector result(v.size());
    for (int i = 0; i < v.size(); i++) 
    {
        result.v[i] = v[i] - other.v[i];
    }

    return result;    
}

/**
 * @fn double DataVector::operator*(const DataVector &other)
 * @brief Dot product operator for the DataVector class.
 * @param other The DataVector to calculate the dot product with.
 * @return The dot product.
 */
double DataVector::operator*(const DataVector &other)
{
    double product = 0.0;
    for(int i = 0; i < v.size(); i++)
    {
        product += v[i] * other.v[i];
    }

    return product;
}

/**
 * @fn void DataVector::input(double component)
 * @brief Adds a component to the DataVector.
 * @param component The component to add.
 */
void DataVector::input(double component)
{
    v.push_back(component);
}

/**
 * @fn double DataVector::norm()
 * @brief Calculates the norm (magnitude) of the DataVector.
 * @return The norm of the DataVector.
 */
double DataVector::norm()
{
    double magnitude = 0.0;
    for(int i = 0; i < v.size(); i++)
    {
        magnitude += (v[i] * v[i]);
    }

    return sqrt(magnitude);
}

double DataVector::get_element(int j)
{
    return v[j];
}

int DataVector::get_the_size()
{
    return v.size();
}

/**
 * @fn void DataVector::print_vector()
 * @brief Prints the components of the DataVector.
 */
void DataVector::print_vector()
{
    for(int i = 0; i < v.size(); i++)
    {
        printf("%.2lf ", v[i]);
    }
    printf("\n");
}

void DataVector::random_vector(int dimension)
{
    double magnitude = 0;
    for(int i = 0; i < dimension; i++)
    {
        int temp = (rand()%1000)/10 ;
        v.push_back(temp);
        magnitude += temp*temp;
    }
    magnitude = sqrt(magnitude);

    for(int i=0; i < dimension; i++)
    {
        v[i] = v[i] / magnitude;
    }
}

/**
 * @fn void VectorDataset::print_datavector()
 * @brief Prints the vectors in the dataset.
 */
VectorDataset::VectorDataset()
{

}

/**
 * @fn VectorDataset::~VectorDataset()
 * @brief Destroys the VectorDataset and frees the memory.
 */
VectorDataset::~VectorDataset()
{
    vector<DataVector>().swap(v);
}

/**
 * @fn VectorDataset & VectorDataset::operator=(const VectorDataset &other)
 * @brief Assigns the values from another VectorDataset.
 * @param other The other VectorDataset.
 * @return A reference to the VectorDataset.
 */
VectorDataset & VectorDataset::operator=(const VectorDataset &other)
{
    v = other.v;
    return *this;
}

/**
 * @fn void VectorDataset::ReadDataset()
 * @brief Reads the dataset from a file.
 */
void VectorDataset::ReadDataset()
{
    ifstream file("fmnist-train.csv");

    if(file.is_open())
    {
        string line;
        while(getline(file, line))
        {
            DataVector temp;
            stringstream ss(line);
            string value;

            while(getline(ss, value, ','))
            {
                temp.input(stod(value));
            }
            v.push_back(temp);
        }

        file.close();
    }
    else
    {
        printf("File not found\n");
    }
}

/**
 * @fn int VectorDataset::row_size()
 * @brief Gets the size of the dataset.
 * @return The size of the dataset.
 */
int VectorDataset::row_size()
{
    return v.size();
}

/**
 * @fn DataVector VectorDataset::access_row(int i)
 * @brief Accesses a vector at a given index.
 * @param i The index.
 * @return The vector.
 */
DataVector VectorDataset::access_row(int i)
{
    return v[i];
}

double VectorDataset::access_element(int i, int j)
{
    return v[i].get_element(j);
}

/**
 * @fn void VectorDataset::add_vector(DataVector vec)
 * @brief Adds a vector to the dataset.
 * @param vec The vector to add.
 */
void VectorDataset::add_vector(DataVector vec)
{
    v.push_back(vec);
}

void VectorDataset::erase_vector(int i)
{
    v.erase(v.begin() + i);
}

/**
 * @fn void VectorDataset::print_datavector()
 * @brief Prints the vectors in the dataset.
 */
void VectorDataset::print_datavector()
{
    for(int i = 0; i < v.size(); i++)
    {
        v[i].print_vector();
    }
}

TreeIndex::TreeIndex()
{
    D.ReadDataset();
}

struct kd_tree_node* KDTreeIndex::new_kd_node(vector<int>* a, int h)
{
    // Allocating memory for a new node
    struct kd_tree_node* temp = new kd_tree_node();

    // Sending the indices of this node into it's respective position
    for(int i=0; i<a->size(); i++)
    {
        temp->indices.push_back(a->at(i));
    }

    // The height of this node is h
    temp->height = h;

    // Creating a duplicate vector to sort based on their h%maxcols
    vector<double>* temp_vector = new vector<double>();

    // Populating the temp_vector with the hth dimension of the indices
    for(int i=0; i<a->size(); i++)
    {
        temp_vector->push_back(D.access_element(a->at(i), h%max_cols));
    }

    // Sorting the temp_vector
    sort(temp_vector->begin(), temp_vector->end());

    // Finding the median of the hth dimension
    if(temp_vector->size() % 2 == 0)
    {
        temp->median = (temp_vector->at(temp_vector->size()/2) + temp_vector->at(temp_vector->size()/2 - 1)) / 2;
    }
    else
    {
        temp->median = temp_vector->at(temp_vector->size()/2);
    }

    vector<int>* temp_left = new vector<int>();
    vector<int>* temp_right = new vector<int>();

    // Populating the left and right vectors based on the median
    for(int i=0; i<a->size(); i++)
    {
        if(D.access_element(a->at(i), h%max_cols) <= temp->median)
        {
            temp_left->push_back(a->at(i));
        }
        else
        {
            temp_right->push_back(a->at(i));
        }
    }

    // If the left vector is empty, then the left node is NULL
    if(temp_left->empty())
    {
        temp->left = NULL;
    }
    // If the left vector has only one element, then the left node is a leaf node
    else if(temp_left->size() == 1)
    {
        struct kd_tree_node* templ = new kd_tree_node();
        templ->indices.insert(templ->indices.end(), temp_left->begin(), temp_left->end());
        templ->height = temp->height + 1;
        templ->median = D.access_element(templ->indices[0], h%max_cols);

        templ->left = NULL;
        templ->right = NULL;

        temp->left = templ; 
    }
    else
    {
        // If the left vector has more than one element, then the left node is a new node
        temp->left = new_kd_node(temp_left, temp->height + 1);
    }

    // If the right vector is empty, then the right node is NULL
    if(temp_right->empty())
    {
        temp->right = NULL;
    }
    // If the right vector has only one element, then the right node is a leaf node
    else if(temp_right->size() == 1)
    {
        struct kd_tree_node* tempr = new kd_tree_node();
        tempr->indices.insert(tempr->indices.end(), temp_right->begin(), temp_right->end());
        tempr->height = temp->height + 1;
        tempr->median = D.access_element(tempr->indices[0], h%max_cols);

        tempr->left = NULL;
        tempr->right = NULL;

        temp->right = tempr; 
    }
    else
    {
        // If the right vector has more than one element, then the right node is a new node
        temp->right = new_kd_node(temp_right, temp->height + 1);
    }

    delete temp_vector;
    delete temp_right;
    delete temp_left;

    return temp;
}

void KDTreeIndex::print_kd_tree(struct kd_tree_node* head)
{
    if(head == NULL)
    {
        return;
    }

    printf("Height: %d\n", head->height);
    printf("Median: %.2lf\n", head->median);
    printf("Indices: ");
    for(int i=0; i<head->indices.size(); i++)
    {
        printf("%d ", head->indices[i]);
    }
    printf("\n\n");

    print_kd_tree(head->left);
    print_kd_tree(head->right);
}

KDTreeIndex::KDTreeIndex()
{
    auto start = chrono::high_resolution_clock::now();

    // Creating a root node
    struct kd_tree_node* temp = new kd_tree_node();

    // Sending the all the indices in the DataSet
    for(int i=0; i<D.row_size(); i++)
    {
        temp->indices.push_back(i);
    }

    // Height 0 since it is a root node
    temp->height = 0;

    // Sorting the indices based on the first dimension
    vector<double>* temp_vector = new vector<double>();

    for(int i=0; i<D.row_size(); i++)
    {
        temp_vector->push_back(D.access_element(i, 0));
    }

    sort(temp_vector->begin(), temp_vector->end());

    // Finding the medain of the nth-dimension
    if(temp_vector->size() % 2 == 0)
    {
        temp->median = (temp_vector->at(temp_vector->size()/2) + temp_vector->at(temp_vector->size()/2 - 1)) / 2;
    }
    else
    {
        temp->median = temp_vector->at(temp_vector->size()/2);
    }

    vector<int>* temp_left = new vector<int>();
    vector<int>* temp_right = new vector<int>();

    for(int i=0; i<D.row_size(); i++)
    {
        if(D.access_element(temp->indices[i], 0) <= temp->median)
        {
            temp_left->push_back(temp->indices[i]);
        }
        else
        {
            temp_right->push_back(temp->indices[i]);
        }
    }

    if(temp_left->empty())
    {
        temp->left = NULL;
    }
    else if(temp_left->size() == 1)
    {
        struct kd_tree_node* templ = new kd_tree_node();
        templ->indices.insert(templ->indices.end(), temp_left->begin(), temp_left->end());
        templ->height = temp->height + 1;
        templ->median = D.access_element(templ->indices[0], templ->height);

        templ->left = NULL;
        templ->right = NULL;

        temp->left = templ; 
    }
    else
    {
        temp->left = new_kd_node(temp_left, temp->height + 1);
    }

    if(temp_right->empty())
    {
        temp->right = NULL;
    }
    else if(temp_right->size() == 1)
    {
        struct kd_tree_node* tempr = new kd_tree_node();
        tempr->indices.insert(tempr->indices.end(), temp_right->begin(), temp_right->end());
        tempr->height = temp->height + 1;
        tempr->median = D.access_element(tempr->indices[0], tempr->height);

        tempr->left = NULL;
        tempr->right = NULL;

        temp->right = tempr; 
    }
    else
    {
        temp->right = new_kd_node(temp_right, temp->height + 1);
    }
    
    root = temp;
    printf("\nKD-Tree successfully built\n");
    delete temp_vector;
    delete temp_left;
    delete temp_right;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("Time taken to build KD-Tree: %ld ms\n\n", duration.count());
}

TreeIndex* TreeIndex::instance = nullptr;
KDTreeIndex* KDTreeIndex::kdinstance = nullptr;
RPTreeIndex* RPTreeIndex::rpinstance = nullptr;

void delete_kd_tree(struct kd_tree_node*& head)
{
    if(head == NULL)
    {
        return;
    }

    delete_kd_tree(head->left);
    delete_kd_tree(head->right);
    delete head;

    head = NULL;
}

void KDTreeIndex::add_kd_vector(DataVector temp)
{
    int d = temp.get_the_size();

    if(d > max_cols)
    {
        printf("Dimension exceeds the maximum dimension\n");
        return;
    }

    TreeIndex::GetInstance().add_datavector(temp);

    ofstream file("fmnist-train.csv", ios::app);
    if (!file.is_open()) {
        cout << "Failed to open the file." << endl;
        return; // Exit if file not opened successfully
    }

    // Write the elements of the vector to the file
    for (int i = 0; i < d; i++) {
        if(i != max_cols-1) file << fixed << setprecision(1) << temp.get_element(i) << ",";
        else file << fixed << setprecision(1) << temp.get_element(i); 
    }

    // Fill the remaining positions with 0.0 if the vector size is less than max_cols
    for (int i = d; i < max_cols; i++) {
        if(i != max_cols-1) file << fixed << setprecision(1) << 0.0 << ",";
        else file << fixed << setprecision(1) << 0.0;
    }
    file << "\n"; // Add a newline character at the end of the line

    // Close the file
    file.close();

    delete_kd_tree(KDTreeIndex::root);

    KDTreeIndex::kdinstance = nullptr;
    printf("KD-Tree successfully updated on addition\n");

}

void KDTreeIndex::delete_kd_vector(int d)
{

    if(d > D.row_size())
    {
        printf("Index exceeds the maximum index\n");
        return;
    }

    D.erase_vector(d);

    ofstream file("fmnist-train.csv", ios::trunc);
    
    if (!file.is_open()) {
        cout << "Failed to open the file." << endl;
        return; // Exit if file not opened successfully
    }

    // Write the elements of the vector to the file
    for (int i = 0; i < D.row_size(); i++) {
        for (int j = 0; j < D.access_row(i).get_the_size(); j++) {
            if(j != max_cols-1) file << fixed << setprecision(1) << D.access_element(i, j) << ",";
            else file << fixed << setprecision(1) << D.access_element(i, j);
        }
        file << "\n"; // Add a newline character at the end of the line
    }

    // Close the file
    file.close();

    delete_kd_tree(KDTreeIndex::root);

    KDTreeIndex::kdinstance = nullptr;
    printf("KD-Tree successfully updated after deletion\n");
}

void RPTreeIndex::print_rp_tree(struct rp_tree_node* head)
{
    if(head == NULL)
    {
        return;
    }

    printf("Height: %d\n", head->height);
    printf("Median: %.2lf\n", head->median);
    printf("Median Vector: ");
    head->median_vector.print_vector();
    printf("Indices: ");
    for(int i=0; i<head->indices.size(); i++)
    {
        printf("%d ", head->indices[i]);
    }
    printf("\n\n");

    print_rp_tree(head->left);
    print_rp_tree(head->right);
}

struct rp_tree_node* RPTreeIndex::new_rp_node(vector<int>* a, int h)
{
    // Allocating memory for a new node
    struct rp_tree_node* temp = new rp_tree_node();

    // Sending the indices of this node into it's respective postion
    for(int i=0; i<a->size(); i++)
    {
        temp->indices.push_back(a->at(i));
    }

    // The height of this node is h
    temp->height = h;

    // Allocating the random vector
    temp->median_vector.random_vector(max_cols);

    // Temporary vector to store the dot products
    vector<double>* temp_vector = new vector<double>();

    // Populating the vector with all the dot products
    for(int i = 0; i < a->size(); i++)
    {
        temp_vector->push_back(D.access_row(a->at(i)) * temp->median_vector);
    }

    // Sorting the temp_vector
    sort(temp_vector->begin(), temp_vector->end());

    // Finding the median
    if(temp_vector->size() % 2 == 0)
    {
        temp->median = (temp_vector->at(temp_vector->size()/2) + temp_vector->at(temp_vector->size()/2 - 1)) / 2;
    }
    else
    {
        temp->median = temp_vector->at(temp_vector->size()/2);
    }

    vector<int>* temp_left = new vector<int>();
    vector<int>* temp_right = new vector<int>();

    for(int i = 0; i < a->size(); i++)
    {
        if(D.access_row(a->at(i)) * temp->median_vector <= temp->median)
        {
            temp_left->push_back(a->at(i));
        }
        else
        {
            temp_right->push_back(a->at(i));
        }
    }

    if(temp_left->empty())
    {
        temp->left = NULL;
    }
    else if(temp_left->size() == 1)
    {
        struct rp_tree_node* templ = new rp_tree_node();
        templ->indices.insert(templ->indices.end(), temp_left->begin(), temp_left->end());
        templ->height = temp->height + 1;
        templ->median_vector.random_vector(max_cols);
        templ->median = (D.access_row(templ->indices[0]) * templ->median_vector);

        templ->left = NULL;
        templ->right = NULL;

        temp->left = templ;
    }
    else
    {
        temp->left = new_rp_node(temp_left, temp->height + 1);
    }
    

    if(temp_right->empty())
    {
        temp->right = NULL;
    }
    else if(temp_right->size() == 1)
    {
        struct rp_tree_node* tempr = new rp_tree_node();
        tempr->indices.insert(tempr->indices.end(), temp_right->begin(), temp_right->end());
        tempr->height = temp->height + 1;
        tempr->median_vector.random_vector(max_cols);
        tempr->median = (D.access_row(tempr->indices[0]) * tempr->median_vector);

        tempr->left = NULL;
        tempr->right = NULL;

        temp->right = tempr;
    }
    else
    {
        temp->right = new_rp_node(temp_right, temp->height + 1);
    }

    delete temp_vector;
    delete temp_right;
    delete temp_left;

    return temp;    
}

RPTreeIndex::RPTreeIndex()
{
    auto start = chrono::high_resolution_clock::now();

    // Creating a root node
    struct rp_tree_node* temp = new rp_tree_node();

    // Sending the all the indices in the DataSet
    for(int i=0; i<D.row_size(); i++)
    {
        temp->indices.push_back(i);
    }

    // Height is 0 since it the root
    temp->height = 0;

    temp->median_vector.random_vector(max_cols);

    vector<double>* temp_vector = new vector<double>();

    for(int i = 0; i < D.row_size(); i++)
    {
        temp_vector->push_back((D.access_row(i) * temp->median_vector));
    }

    sort(temp_vector->begin(), temp_vector->end());

    // Finding the median
    if(temp_vector->size() % 2 == 0)
    {
        temp->median = (temp_vector->at(temp_vector->size()/2) + temp_vector->at(temp_vector->size()/2 - 1)) / 2;
    }
    else
    {
        temp->median = temp_vector->at(temp_vector->size()/2);
    }

    vector<int>* temp_left = new vector<int>();
    vector<int>* temp_right = new vector<int>();

    for(int i = 0; i< D.row_size(); i++)
    {
        if((D.access_row(i) * temp->median_vector) <= temp->median)
        {
            temp_left->push_back(temp->indices[i]);
        }
        else
        {
            temp_right->push_back(temp->indices[i]);
        }
    }

    if(temp_left->empty())
    {
        temp->left = NULL;
    }
    else if(temp_left->size() == 1)
    {
        struct rp_tree_node* templ = new rp_tree_node();
        templ->indices.insert(templ->indices.end(), temp_left->begin(), temp_left->end());
        templ->height = temp->height + 1;
        templ->median_vector.random_vector(max_cols);
        templ->median = (D.access_row(templ->indices[0]) * templ->median_vector);

        templ->left = NULL;
        templ->right = NULL;

        temp->left = templ;
    }
    else
    {
        temp->left = new_rp_node(temp_left, temp->height + 1);
    }
    

    if(temp_right->empty())
    {
        temp->right = NULL;
    }
    else if(temp_right->size() == 1)
    {
        struct rp_tree_node* tempr = new rp_tree_node();
        tempr->indices.insert(tempr->indices.end(), temp_right->begin(), temp_right->end());
        tempr->height = temp->height + 1;
        tempr->median_vector.random_vector(max_cols);
        tempr->median = (D.access_row(tempr->indices[0]) * tempr->median_vector);

        tempr->left = NULL;
        tempr->right = NULL;

        temp->right = tempr;
    }
    else
    {
        temp->right = new_rp_node(temp_right, temp->height + 1);
    }
     
    root = temp;
    printf("RP-Tree successfully built\n");
    delete temp_vector;
    delete temp_left;
    delete temp_right;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("Time taken to build RP-Tree: %ld ms\n\n", duration.count());
}

void delete_rp_tree(struct rp_tree_node*& head)
{
    if(head == NULL)
    {
        return;
    }

    delete_rp_tree(head->left);
    delete_rp_tree(head->right);
    delete head;

    head = NULL;
}

void RPTreeIndex::add_rp_vector(DataVector temp)
{
    int d = temp.get_the_size();

    if(d > max_cols)
    {
        printf("Dimension exceeds the maximum dimension\n");
        return;
    }

    TreeIndex::GetInstance().add_datavector(temp);

    ofstream file("fmnist-train.csv", ios::app);
    if (!file.is_open()) {
        cout << "Failed to open the file." << endl;
        return; // Exit if file not opened successfully
    }

    // Write the elements of the vector to the file
    for (int i = 0; i < d; i++) {
        if(i != max_cols-1) file << fixed << setprecision(1) << temp.get_element(i) << ",";
        else file << fixed << setprecision(1) << temp.get_element(i); 
    }

    // Fill the remaining positions with 0.0 if the vector size is less than max_cols
    for (int i = d; i < max_cols; i++) {
        if(i != max_cols-1) file << fixed << setprecision(1) << 0.0 << ",";
        else file << fixed << setprecision(1) << 0.0;
    }
    file << "\n"; // Add a newline character at the end of the line

    // Close the file
    file.close();

    delete_rp_tree(RPTreeIndex::root);

    RPTreeIndex::rpinstance = nullptr;
    printf("RP-Tree successfully updated on addition\n");
}

void RPTreeIndex::delete_rp_vector(int d)
{

    if(d > D.row_size())
    {
        return;
    }

    // D.erase_vector(d);

    // ofstream file("fmnist-train.csv", ios::trunc);
    
    // if (!file.is_open()) {
    //     cout << "Failed to open the file." << endl;
    //     return; // Exit if file not opened successfully
    // }

    // // Write the elements of the vector to the file
    // for (int i = 0; i < D.row_size(); i++) {
    //     for (int j = 0; j < D.access_row(i).get_the_size(); j++) {
    //         if(j != max_cols-1) file << fixed << setprecision(1) << D.access_element(i, j) << ",";
    //         else file << fixed << setprecision(1) << D.access_element(i, j);
    //     }
    //     file << "\n"; // Add a newline character at the end of the line
    // }

    // // Close the file
    // file.close();

    delete_rp_tree(RPTreeIndex::root);

    RPTreeIndex::rpinstance = nullptr;
    printf("RP-Tree successfully updated after deletion\n");
}

void KDTreeIndex::kd_neighbours(int k, DataVector q, int count)
{
    struct kd_tree_node* head = KDTreeIndex::GetInstance().get_root();

    if(head->indices.size() <k)
    {
        printf("There are only %d vectors in the dataset\n", head->indices.size());
        printf("Therefore the %d nearest neighbours are :-\n", head->indices.size());

        for(int i=0; i<head->indices.size(); i++)
        {
            D.access_row(head->indices[i]).print_vector();
        }
    }
    else if(head->indices.size() == k)
    {
        printf("The %d nearest neighbours are :-\n", k);
        for(int i=0; i<head->indices.size(); i++)
        {
            D.access_row(head->indices[i]).print_vector();
        }
    }
    {
        // Priority queue for the k nearest neighbors
        priority_queue<pair<double, int>> nearest_neighbors;

        // Set for the indices already added to the priority queue
        set<int> added_indices;

        // Stack for the nodes to visit
        stack<kd_tree_node*> nodes_to_visit;
        nodes_to_visit.push(head);

        while(!nodes_to_visit.empty())
        {
            kd_tree_node* temp = nodes_to_visit.top();
            nodes_to_visit.pop();

            int split_dimension = temp->height % max_cols;

            // Check if the current node's vectors are closer
            for(int i = 0; i < temp->indices.size(); i++)
            {
                if(added_indices.count(temp->indices[i]) > 0)
                {
                    // This index has already been added to the priority queue
                    continue;
                }

                double distance = (D.access_row(temp->indices[i]) - q).norm();
                if(nearest_neighbors.size() < k || distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.push(make_pair(distance, temp->indices[i]));
                    added_indices.insert(temp->indices[i]);
                    if(nearest_neighbors.size() > k)
                    {
                        nearest_neighbors.pop();
                    }
                }
            }

            // Decide which child node to visit first
            kd_tree_node* first = temp->left;
            kd_tree_node* second = temp->right;
            if(q.get_element(split_dimension) > temp->median)
            {
                swap(first, second);
            }

            // Visit the first child node
            if(first != nullptr)
            {
                nodes_to_visit.push(first);
            }

            // Check if we need to visit the second child node
            if(second != nullptr && (nearest_neighbors.size() < k || 
                abs(q.get_element(split_dimension) - temp->median) < nearest_neighbors.top().first))
            {
                nodes_to_visit.push(second);
            }
        }

        // Print the k nearest neighbors
        printf("The %d nearest neighbours of vector with index %d are :-\n", k, count);
        while(!nearest_neighbors.empty())
        {
            printf("Distance: %.2lf \nVector: \n", nearest_neighbors.top().first);
            D.access_row(nearest_neighbors.top().second).print_vector();
            printf(" ------------------------------ \n");
            nearest_neighbors.pop();
        }
    }
}

void KDTreeIndex::knn_kd()
{
    int k;
    printf("Enter the value of k\n");
    cin >> k;

    int i =0;
    ifstream file("fmnist-test.csv");

    if(file.is_open())
    {
        printf("File opened successfully\n");
        string line;

        auto start = chrono::high_resolution_clock::now();
        
        while(getline(file, line))
        {
            DataVector temp;
            stringstream ss(line);
            string value;

            while(getline(ss, value, ','))
            {
                temp.input(stod(value));
            }

            kd_neighbours(k, temp, i);
            i++;
            printf(" ===========================\n===========================\n\n");
        }
        file.close();

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        printf("Time taken to find the nearest neighbours using KD-Tree is: %ld ms\n\n", duration.count());
    }
    else printf("File not found !!\n");
}

void RPTreeIndex::rp_neighbours(int k, DataVector q, int count)
{
        struct rp_tree_node* head = RPTreeIndex::GetInstance().get_root();

    if(head->indices.size() <k)
    {
        printf("There are only %d vectors in the dataset\n", head->indices.size());
        printf("Therefore the %d nearest neighbours are :-\n", head->indices.size());

        for(int i=0; i<head->indices.size(); i++)
        {
            D.access_row(head->indices[i]).print_vector();
        }
    }
    else if(head->indices.size() == k)
    {
        printf("The %d nearest neighbours are :-\n", k);
        for(int i=0; i<head->indices.size(); i++)
        {
            D.access_row(head->indices[i]).print_vector();
        }
    }
    else
    {
        // Priority queue for the k nearest neighbors
        priority_queue<pair<double, int>> nearest_neighbors;

        // Set for the indices already added to the priority queue
        set<int> added_indices;

        // Stack for the nodes to visit
        stack<rp_tree_node*> nodes_to_visit;
        nodes_to_visit.push(head);

        while(!nodes_to_visit.empty())
        {
            rp_tree_node* temp = nodes_to_visit.top();
            nodes_to_visit.pop();

            // Check if the current node's vectors are closer
            for(int i = 0; i < temp->indices.size(); i++)
            {
                if(added_indices.count(temp->indices[i]) > 0)
                {
                    // This index has already been added to the priority queue
                    continue;
                }

                double distance = (D.access_row(temp->indices[i]) - q).norm();
                if(nearest_neighbors.size() < k || distance < nearest_neighbors.top().first)
                {
                    nearest_neighbors.push(make_pair(distance, temp->indices[i]));
                    added_indices.insert(temp->indices[i]);
                    if(nearest_neighbors.size() > k)
                    {
                        nearest_neighbors.pop();
                    }
                }
            }

            // Decide which child node to visit first
            rp_tree_node* first = temp->left;
            rp_tree_node* second = temp->right;
            if((temp->median_vector * q) < temp->median)
            {
                swap(first, second);
            }

            // Visit the first child node
            if(first != nullptr)
            {
                nodes_to_visit.push(first);
            }

            // Check if we need to visit the second child node
            if(second != nullptr && (nearest_neighbors.size() < k || 
                abs((temp->median_vector * q) - temp->median) < nearest_neighbors.top().first))
            {
                nodes_to_visit.push(second);
            }
        }

        // Print the k nearest neighbors
        printf("The %d nearest neighbours of vector with index %d are :-\n", k, count);

        while(!nearest_neighbors.empty())
        {
            printf("Distance: %.2lf\n Vector: \n", nearest_neighbors.top().first);
            D.access_row(nearest_neighbors.top().second).print_vector();
            printf(" ------------------------------ \n");
            nearest_neighbors.pop();
        }
    }
}

void RPTreeIndex::knn_rp()
{
    int k;
    printf("Enter the value of k\n");
    cin >> k;

    int i =0;
    ifstream file("fmnist-test.csv");

    if(file.is_open())
    {
        printf("File opened successfully\n");
        string line;

        auto start = chrono::high_resolution_clock::now();
        
        while(getline(file, line))
        {
            DataVector temp;
            stringstream ss(line);
            string value;

            while(getline(ss, value, ','))
            {
                temp.input(stod(value));
            }

            rp_neighbours(k, temp, i);
            i++;
            printf(" ===========================\n===========================\n\n");
        }
        file.close();

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        printf("Time taken to find the nearest neighbours using RP-Tree is: %ld ms\n\n", duration.count());
    }
    else printf("File not found !!\n");
}

int main(){
    srand(time(NULL));
    int ans = 1;

    while(ans)
    {
        int choice;
        printf("Enter your choice:-\n1) ==> Make the Kd and RP Tree\n2) ==> Add a vector to the dataset\n3) ==> Delete a vector from the dataset\n4) ==> Find the nearest neighbours using KD-Tree\n5) ==> Find the nearest neighbours using RP-Tree\n0) ==> Exit\n");
        scanf("%d", &choice);

        if(choice == 1)
        {
            KDTreeIndex::GetInstance();
            RPTreeIndex::GetInstance();
        }
        else if(choice == 2)
        {
            printf("Enter the dimension of the vector you want to add\n");
            int d;
            scanf("%d", &d);

            if(d>max_cols)
            {
                printf("Dimension exceeds the maximum dimension\n");
            }

            printf("Enter the new vector\n");
            DataVector temp(0);

            for(int i=0; i<d; i++)
            {
                double x;
                cin >> x;
                temp.input(x);
            }

            KDTreeIndex::GetInstance().add_kd_vector(temp);
            RPTreeIndex::GetInstance().add_rp_vector(temp);
        }
        else if(choice == 3)
        {
            int serial_no;
            printf("Enter the index of the vector you want to delete\n");
            cin >> serial_no;
            KDTreeIndex::GetInstance().delete_kd_vector(serial_no);
            RPTreeIndex::GetInstance().delete_rp_vector(serial_no);
        }
        else if(choice == 4)
        {
            KDTreeIndex::GetInstance().knn_kd();
        }
        else if(choice == 5)
        {
            RPTreeIndex::GetInstance().knn_rp();
        }
        else if(choice == 0)
        {
            break;
        }
    }

    return 0;
}

