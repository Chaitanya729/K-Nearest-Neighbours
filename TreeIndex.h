#include <bits/stdc++.h>

using namespace std;

/**
 * @class DataVector
 * @brief This class represents a vector of doubles.
 */
typedef class DataVector 
{ 

vector<double> v; 

public: 
    /**
     * @fn DataVector::DataVector(int dimension=0)
     * @brief Constructor for the DataVector class.
     * @param dimension The dimension of the vector. Default is 0.
     */
    DataVector(int dimension=0);                            // constructor
    
    /**
     * @fn DataVector::~DataVector()
     * @brief Destructor for the DataVector class.
     */
    ~DataVector(); // destructor    
    
    /**
     * @fn DataVector::DataVector(const DataVector& other)
     * @brief Copy constructor for the DataVector class.
     * @param other The DataVector to copy.
     */
    DataVector(const DataVector& other);                    // copy constructor

    /**
     * @fn DataVector & DataVector::operator=(const DataVector &other)
     * @brief Assignment operator for the DataVector class.
     * @param other The DataVector to assign from.
     * @return A reference to this DataVector.
     */
    DataVector & operator=(const DataVector &other);        // assignment operator 
    
    /**
     * @fn void DataVector::setDimension(int dimension)
     * @brief Sets the dimension of the DataVector.
     * @param dimension The new dimension.
     */
    void setDimension(int dimension);                       // set dimension
    
    /**
     * @fn DataVector DataVector::operator+(const DataVector &other)
     * @brief Addition operator for the DataVector class.
     * @param other The DataVector to add.
     * @return The resulting DataVector.
     */
    DataVector operator+(const DataVector &other);          // addition operator
    
    /**
     * @fn DataVector DataVector::operator-(const DataVector &other)
     * @brief Subtraction operator for the DataVector class.
     * @param other The DataVector to subtract.
     * @return The resulting DataVector.
     */
    DataVector operator-(const DataVector &other);          // subtraction operator
    
    /**
     * @fn double DataVector::operator*(const DataVector &other)
     * @brief Dot product operator for the DataVector class.
     * @param other The DataVector to calculate the dot product with.
     * @return The dot product.
     */
    double operator*(const DataVector &other);              // dot product operator

    /**
     * @fn void DataVector::input(double component)
     * @brief Adds a component to the DataVector.
     * @param component The component to add.
     */
    void input(double component);

    /**
     * @fn double DataVector::norm()
     * @brief Calculates the norm (magnitude) of the DataVector.
     * @return The norm of the DataVector.
     */
    double norm();                                          // norm of the vector
    
    double get_element(int j);

    int get_the_size();

    void random_vector(int dimension);

    /**
     * @fn void DataVector::print_vector()
     * @brief Prints the components of the DataVector.
     */
    void print_vector();

} DataVector;

/**
 * @class VectorDataset
 * @brief A class to represent a dataset of vectors.
 * @var VectorDataset::v
 * @brief The vectors in the dataset.
 */
typedef class VectorDataset{

    vector<DataVector> v;

    public:
        /**
         * @fn VectorDataset::VectorDataset()
         * @brief Constructs a new VectorDataset.
         */
        VectorDataset();

        /**
         * @fn VectorDataset::~VectorDataset()
         * @brief Destroys the VectorDataset.
         */
        ~VectorDataset();

        /**
         * @fn VectorDataset & VectorDataset::operator=(const VectorDataset &other)
         * @brief Assigns the values from another VectorDataset.
         * @param other The other VectorDataset.
         * @return A reference to the VectorDataset.
         */
        VectorDataset & operator=(const VectorDataset &other);

        /**
         * @fn void VectorDataset::ReadDataset()
         * @brief Reads the dataset from a file.
         */
        void ReadDataset();

        /**
         * @fn int VectorDataset::row_size()
         * @brief Gets the size of the dataset.
         * @return The size of the dataset.
         */
        int row_size();

        /**
         * @fn DataVector VectorDataset::access_row(int i)
         * @brief Accesses a vector at a given index.
         * @param i The index.
         * @return The vector.
         */
        DataVector access_row(int i);

        double access_element(int i, int j);

        /**
         * @fn void VectorDataset::add_vector(DataVector vec)
         * @brief Adds a vector to the dataset.
         * @param vec The vector to add.
         */
        void add_vector(DataVector vec);

        void erase_vector(int i);

        /**
         * @fn void VectorDataset::print_datavector()
         * @brief Prints the vectors in the dataset.
         */
        void print_datavector();

} VectorDataset;

struct kd_tree_node
{
    vector<int> indices;
    int height;
    double median;

    kd_tree_node* left;
    kd_tree_node* right;
};

struct rp_tree_node
{
    vector<int> indices;
    int height;
    DataVector median_vector;
    double median;

    rp_tree_node* left;
    rp_tree_node* right;
};

class TreeIndex
{
    static TreeIndex *instance;
protected:
    VectorDataset D;
    TreeIndex();
    
public:
    static TreeIndex &GetInstance()
    {
        if(instance == NULL)
        {
            instance = new TreeIndex();
        }
        return *instance;
    }

    void add_datavector(DataVector vec)
    {
        D.add_vector(vec);
    }
};

class KDTreeIndex : public TreeIndex
{
    struct kd_tree_node* root;
    static KDTreeIndex *kdinstance;
public:
    static KDTreeIndex &GetInstance()
    {
        if(kdinstance == NULL)
        {
            kdinstance = new KDTreeIndex();
        }
        return *kdinstance;
    }

    struct kd_tree_node* get_root()
    {
        return root;
    }

    void print_kd_tree(struct kd_tree_node* head);

    struct kd_tree_node* new_kd_node(vector<int>* a, int h);   

    void add_kd_vector(DataVector temp);

    void delete_kd_vector(int d); 

    void knn_kd();

    void kd_neighbours(int k, DataVector q, int count);

private:
    KDTreeIndex();
};

class RPTreeIndex : public TreeIndex
{
    struct rp_tree_node* root;
    static RPTreeIndex *rpinstance;
public:
    static RPTreeIndex &GetInstance()
    {
        if(rpinstance == NULL)
        {
            rpinstance = new RPTreeIndex();
        }
        return *rpinstance;
    }

    struct rp_tree_node* new_rp_node(vector<int>* a, int h);

    void print_rp_tree(struct rp_tree_node* head);

    struct rp_tree_node* get_root()
    {
        return root;
    }

    void add_rp_vector( DataVector temp);

    void delete_rp_vector(int d);

    void knn_rp();

    void rp_neighbours(int k, DataVector q, int count);

private:
    RPTreeIndex();
};